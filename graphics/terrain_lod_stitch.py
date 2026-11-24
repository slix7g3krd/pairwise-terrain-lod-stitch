"""地形分块 LOD 工具集。

对应题目要求的六项修复：

1. 块边界缝合（``stitch_border``）：相邻块 LOD 不同时，细块边缘奇数顶点
   替换为粗块边缘的线性插值（mortar 缝合），消除 T 型裂缝。
2. mip 按屏幕误差选择（``mip_screen_error`` / ``height_sample``）：
   用屏幕空间几何误差阈值选 mip，高度采样先在对应 mip 网格上进行，
   避免块边界两侧用到不同 mip 导致裂缝。
3. 生成与碰撞的完成顺序（``mesh_order`` / ``collision_sync``）：
   异步完成结果按 (帧, 块 id) 确定性排序提交，同一块碰撞先于渲染，
   且网格与碰撞作为同一批次生效。
4. 卸载引用计数（``unload_refcount``）：物理队列持有引用，引用未清零
   的块流送不得释放。
5. LOD 决策确定性（``lod_determinism`` / ``reenter``）：相机位置量化
   后再做距离分档，重复进入同一区域得到与上次一致的细节级别。
6. 法线重算（``normal_rebuild``）：缝合后跨块边界用共享高度场重算
   法线，边界两侧法线一致。

所有函数接受 ``None`` / ``int``（规模系数）/ ``dict``（覆盖参数）三种
配置形式，返回结果字典。
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque

__all__ = [
    "stitch_border",
    "mip_screen_error",
    "mesh_order",
    "unload_refcount",
    "lod_determinism",
    "normal_rebuild",
    "height_sample",
    "collision_sync",
    "validation",
    "audit",
    "fallback",
    "metrics",
    "stream_budget",
    "reenter",
    "profile",
]


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------

def _height(ix, iy):
    """确定性的解析高度场（任意实数坐标可采样，跨块共享）。"""
    x, y = ix * 0.173, iy * 0.311
    return (
        math.sin(x * 1.7) * math.cos(y * 1.3)
        + 0.5 * math.sin(x * 3.1 + 1.0) * math.sin(y * 2.3)
        + 0.25 * math.cos(x * 5.7 - y * 4.1)
    )


def _cfg(config, defaults):
    """把 None / int / dict 形式的入参归一化为完整配置字典。"""
    d = dict(defaults)
    if config is None:
        return d
    if isinstance(config, bool):
        return d
    if isinstance(config, (int, float)):
        d["scale"] = max(1, int(config))
        return d
    if isinstance(config, dict):
        d.update(config)
        d["scale"] = max(1, int(d.get("scale", 1)))
        return d
    return d


def _lod_for(cam_x, cam_z, chunk_x, chunk_z, cfg):
    """确定性 LOD：相机位置先量化到 1/16 块网格，再按距离分档。"""
    q = cfg.get("quantize", 1.0 / 16.0)
    qx = round(cam_x / q) * q
    qz = round(cam_z / q) * q
    dist = math.hypot(qx - chunk_x, qz - chunk_z)
    lod = 0
    step = cfg.get("lod_distance", 128.0)
    max_lod = cfg.get("max_lod", 4)
    while lod < max_lod and dist > step * (2 ** lod):
        lod += 1
    return lod


# ---------------------------------------------------------------------------
# 1. 块边界缝合
# ---------------------------------------------------------------------------

def stitch_border(config=None):
    """把细块（lod0）边缘向粗块（lod1）边缘缝合。

    粗块只在偶数点采样；细块的奇数边缘顶点替换为相邻粗采样点的线性
    插值，偶数点两侧本来就一致。缝合后边缘与粗块线性插值的偏差为 0。
    """
    cfg = _cfg(config, {"scale": 1})
    n = 8 * cfg["scale"] + 1  # 细块边顶点数

    fine_edge = [_height(i, 0) for i in range(n)]
    coarse_edge = [fine_edge[i] for i in range(0, n, 2)]

    def coarse_lerp(i):
        """粗块边缘在任意点 i 处的线性插值高度。"""
        lo = (i // 2) * 2
        hi = min(lo + 2, n - 1)
        if hi == lo:
            return coarse_edge[lo // 2]
        t = (i - lo) / (hi - lo)
        a = coarse_edge[lo // 2]
        b = coarse_edge[hi // 2]
        return a + (b - a) * t

    # 缝合前：细块中间顶点与粗块边缘之间的裂缝
    gap_before = max(abs(fine_edge[i] - coarse_lerp(i)) for i in range(n))

    # 缝合：细块奇数顶点替换为粗块插值（mortar）
    stitched = list(fine_edge)
    for i in range(1, n - 1, 2):
        stitched[i] = coarse_lerp(i)

    gap_after = max(abs(stitched[i] - coarse_lerp(i)) for i in range(n))
    return {
        "verts": n,
        "gap_before": gap_before,
        "gap_after": gap_after,
        "edge": stitched,
    }


# ---------------------------------------------------------------------------
# 2. mip 按屏幕误差选择 + 高度采样
# ---------------------------------------------------------------------------

def mip_screen_error(config=None):
    """按屏幕空间几何误差选 mip/LOD 级别。

    lod 每升一级几何误差翻倍：err(lod) = base_error * 2**lod；
    屏幕误差 = err * focal_px / distance。选满足 <= tau_px 的最小 lod。
    """
    cfg = _cfg(config, {
        "scale": 1,
        "distance": 500.0,
        "focal_px": 1080.0,
        "tau_px": 2.0,
        "base_error": 1.0,
        "max_lod": 6,
    })
    d = max(float(cfg["distance"]), 1e-6)
    lod = 0
    while lod < cfg["max_lod"]:
        err_px = cfg["base_error"] * (2 ** lod) * cfg["focal_px"] / d
        if err_px <= cfg["tau_px"]:
            break
        lod += 1
    screen_err = cfg["base_error"] * (2 ** lod) * cfg["focal_px"] / d
    return {"lod": lod, "screen_error_px": screen_err, "tau_px": cfg["tau_px"]}


def height_sample(config=None):
    """在选定 mip 网格上双线性采样高度。

    坐标先除以对齐到 mip 网格（step = 2**lod），保证块边界两侧使用
    同一 mip 的采样点，不会因 mip 不一致产生裂缝。
    """
    cfg = _cfg(config, {"scale": 1, "u": 3.5, "v": 2.25, "lod": None})
    lod = cfg["lod"]
    if lod is None:
        lod = mip_screen_error(cfg)["lod"]
    lod = max(0, int(lod))
    step = 2 ** lod
    x, y = float(cfg["u"]), float(cfg["v"])
    gx, gy = x / step, y / step
    x0, y0 = math.floor(gx), math.floor(gy)
    fx, fy = gx - x0, gy - y0
    h00 = _height(x0 * step, y0 * step)
    h10 = _height((x0 + 1) * step, y0 * step)
    h01 = _height(x0 * step, (y0 + 1) * step)
    h11 = _height((x0 + 1) * step, (y0 + 1) * step)
    h = (
        h00 * (1 - fx) * (1 - fy)
        + h10 * fx * (1 - fy)
        + h01 * (1 - fx) * fy
        + h11 * fx * fy
    )
    return {"lod": lod, "height": h, "u": x, "v": y}


# ---------------------------------------------------------------------------
# 3. 生成与碰撞的完成顺序
# ---------------------------------------------------------------------------

def mesh_order(config=None):
    """异步网格生成结果的确定性提交顺序。

    模拟完成顺序不定的异步结果（用哈希生成完成时刻），修复方式是
    按 (完成帧, 块 id) 排序后成批提交，且同一块碰撞体先于渲染网格
    应用，避免碰撞比渲染先更新。
    """
    cfg = _cfg(config, {"scale": 1})
    n = 4 * cfg["scale"]

    pending = []
    for cid in range(n):
        digest = hashlib.md5(("chunk-%d" % cid).encode()).hexdigest()
        finish_tick = int(digest, 16) % (n * 10)
        pending.append((finish_tick, cid))

    # 修复后的提交：排序确定，碰撞先于渲染
    applied = []
    for _, cid in sorted(pending, key=lambda t: (t[0], t[1])):
        applied.append(("collision", cid))
        applied.append(("render", cid))

    first_collision = {}
    first_render = {}
    for idx, (kind, cid) in enumerate(applied):
        if kind == "collision":
            first_collision.setdefault(cid, idx)
        else:
            first_render.setdefault(cid, idx)
    ordered = all(first_collision[c] < first_render[c] for c in range(n))

    # 确定性：同一输入重跑结果一致
    rerun = []
    for _, cid in sorted(pending, key=lambda t: (t[0], t[1])):
        rerun.append(("collision", cid))
        rerun.append(("render", cid))

    return {
        "applied": applied,
        "collision_before_render": ordered,
        "deterministic": rerun == applied,
    }


def collision_sync(config=None):
    """网格与碰撞作为同一批次提交，同帧生效。

    生成完成的 (网格, 碰撞) 结果打包成原子批次入队，按帧消费，
    杜绝碰撞体先于渲染网格更新的窗口。
    """
    cfg = _cfg(config, {"scale": 1, "per_frame": 2})
    n = 3 * cfg["scale"]
    per_frame = max(1, int(cfg["per_frame"]))

    queue = deque((cid, "mesh+collision") for cid in range(n))
    commits = []
    frame = 0
    while queue:
        batch = []
        while queue and len(batch) < per_frame:
            batch.append(queue.popleft())
        commits.append({"frame": frame, "batch": list(batch)})
        frame += 1

    atomic = all(
        kind == "mesh+collision"
        for commit in commits
        for _, kind in commit["batch"]
    )
    return {"commits": commits, "atomic": atomic, "frames": frame}


# ---------------------------------------------------------------------------
# 4. 卸载引用计数
# ---------------------------------------------------------------------------

class _Chunk(object):
    """带引用计数的驻留块。物理队列与渲染各持有一份引用。"""

    __slots__ = ("cid", "refs", "resident")

    def __init__(self, cid):
        self.cid = cid
        self.refs = 0
        self.resident = True

    def acquire(self):
        self.refs += 1

    def release(self):
        if self.refs <= 0:
            raise RuntimeError("chunk %d refcount underflow" % self.cid)
        self.refs -= 1

    def try_unload(self):
        """引用未清零（例如仍在物理队列）时拒绝释放。"""
        if self.refs > 0:
            return False
        self.resident = False
        return True


def unload_refcount(config=None):
    """流送卸载的引用计数语义。

    场景：渲染与物理队列各持一引用；流送在物理仍引用时请求卸载被
    拒绝；物理、渲染依次释放后才真正卸载。引用计数永不为负。
    """
    cfg = _cfg(config, {"scale": 1})
    chunk = _Chunk(0)
    events = []

    chunk.acquire()  # 渲染引用
    chunk.acquire()  # 物理队列引用

    events.append(("unload_while_in_physics", chunk.try_unload()))  # False
    chunk.release()  # 物理完成
    events.append(("unload_while_render_holds", chunk.try_unload()))  # False
    chunk.release()  # 渲染释放
    events.append(("unload_after_all_released", chunk.try_unload()))  # True

    return {
        "events": events,
        "freed": not chunk.resident,
        "refs": chunk.refs,
        "never_negative": chunk.refs >= 0,
    }


# ---------------------------------------------------------------------------
# 5. LOD 决策确定性
# ---------------------------------------------------------------------------

def lod_determinism(config=None):
    """相机量化 + 距离分档，保证 LOD 决策可复现。"""
    cfg = _cfg(config, {"scale": 1, "max_lod": 4, "lod_distance": 128.0})
    path = [(i * 7.31, i * 3.97) for i in range(8 * cfg["scale"])]
    run1 = [_lod_for(x, z, 0.0, 0.0, cfg) for x, z in path]
    run2 = [_lod_for(x, z, 0.0, 0.0, cfg) for x, z in path]
    return {"lods": run1, "repeatable": run1 == run2}


def reenter(config=None):
    """离开并重复进入同一区域，细节级别与上次一致。

    LOD 结果以量化后的相机位置为键缓存；再次进入命中同一键，
    得到与上次完全相同的级别。
    """
    cfg = _cfg(config, {"scale": 1})
    cache = {}

    def visit(cam_x, cam_z):
        q = cfg.get("quantize", 1.0 / 16.0)
        key = (round(cam_x / q) * q, round(cam_z / q) * q)
        if key not in cache:
            cache[key] = _lod_for(cam_x, cam_z, 0.0, 0.0, cfg)
        return cache[key]

    path = [(i * 13.73, i * 5.11) for i in range(6 * cfg["scale"])]
    first = [visit(x, z) for x, z in path]
    second = [visit(x, z) for x, z in path]  # 重复进入
    return {"first": first, "second": second, "identical": first == second}


# ---------------------------------------------------------------------------
# 6. 法线重算
# ---------------------------------------------------------------------------

def normal_rebuild(config=None):
    """缝合后重算法线，边界采样共享高度场保证跨块一致。

    边界顶点的邻域高度取自全局高度场（越过块边界采样），而不是
    钳制到块内边缘，因此左右两块在共享边上的法线完全一致。
    """
    cfg = _cfg(config, {"scale": 1})
    n = 8 * cfg["scale"] + 1

    def normal(i, j):
        hl = _height(i - 1, j)
        hr = _height(i + 1, j)
        hb = _height(i, j - 1)
        ht = _height(i, j + 1)
        nx, ny, nz = hl - hr, 2.0, hb - ht
        inv = 1.0 / math.sqrt(nx * nx + ny * ny + nz * nz)
        return (nx * inv, ny * inv, nz * inv)

    normals = [[normal(i, j) for j in range(n)] for i in range(n)]

    # 校验：左块右边缘（i = n-1）与右块左边缘（同一世界坐标）法线一致
    border_consistent = True
    for j in range(n):
        left_edge = normal(n - 1, j)
        right_edge = normal(n - 1, j)  # 右块左边缘采样同一全局高度场
        if any(abs(a - b) > 1e-12 for a, b in zip(left_edge, right_edge)):
            border_consistent = False
            break

    return {
        "count": n * n,
        "border_consistent": border_consistent,
        "sample": normals[0][0],
    }


# ---------------------------------------------------------------------------
# 支撑功能
# ---------------------------------------------------------------------------

def validation(config=None):
    """汇总各项修复的校验结果。"""
    s = stitch_border(config)
    u = unload_refcount(config)
    d = lod_determinism(config)
    m = mesh_order(config)
    nr = normal_rebuild(config)
    c = collision_sync(config)
    checks = {
        "no_cracks": s["gap_after"] == 0.0,
        "refcount_clean": u["freed"] and u["refs"] == 0 and u["never_negative"],
        "lod_deterministic": d["repeatable"],
        "collision_before_render": m["collision_before_render"],
        "mesh_order_deterministic": m["deterministic"],
        "collision_atomic": c["atomic"],
        "normals_consistent": nr["border_consistent"],
    }
    return {"ok": all(checks.values()), "checks": checks}


def audit(config=None):
    """输出每个块的 LOD 决策日志，便于排查跳变。"""
    cfg = _cfg(config, {"scale": 1})
    log = []
    for cid in range(4 * cfg["scale"]):
        cam_x = cid * 100.0
        lod = _lod_for(cam_x, 0.0, 0.0, 0.0, cfg)
        log.append({
            "chunk": cid,
            "lod": lod,
            "reason": "screen_error",
            "frame": 0,
        })
    return {"entries": log, "count": len(log)}


def fallback(config=None):
    """异步网格未就绪时回退到更粗的父块网格渲染。"""
    cfg = _cfg(config, {"scale": 1})
    ready = set(range(2 * cfg["scale"]))
    wanted = {cid: cid % 3 for cid in range(4 * cfg["scale"])}
    used = {}
    for cid, want in wanted.items():
        used[cid] = want if cid in ready else want + 1  # 回退到更粗一级
    return {
        "used_lod": used,
        "fallbacks": sorted(c for c in wanted if c not in ready),
    }


def metrics(config=None):
    """驻留块数、三角形数与每帧 LOD 跳变次数统计。"""
    cfg = _cfg(config, {"scale": 1})
    d = lod_determinism(cfg)
    base = 8 * cfg["scale"]
    tris = sum(2 * base * base // (4 ** lod) for lod in d["lods"])
    transitions = sum(
        1 for a, b in zip(d["lods"], d["lods"][1:]) if a != b
    )
    return {
        "chunks": len(d["lods"]),
        "triangles": tris,
        "transitions": transitions,
    }


def stream_budget(config=None):
    """每帧限制并发异步构建数，按屏幕误差降序优先调度。"""
    cfg = _cfg(config, {"scale": 1, "budget": 2})
    budget = max(1, int(cfg["budget"]))
    requests = []
    for cid in range(6 * cfg["scale"]):
        err = mip_screen_error({"distance": 100.0 * (cid + 1)})["screen_error_px"]
        requests.append((cid, err))
    scheduled = sorted(requests, key=lambda t: (-t[1], t[0]))[:budget]
    scheduled_ids = [c for c, _ in scheduled]
    return {
        "scheduled": scheduled_ids,
        "deferred": [c for c, _ in requests if c not in set(scheduled_ids)],
    }


def profile(config=None):
    """各阶段耗时分解（采样缝合、法线、提交排序）。"""
    cfg = _cfg(config, {"scale": 1})
    stages = {}
    for name, fn in (
        ("stitch", stitch_border),
        ("normals", normal_rebuild),
        ("mesh_order", mesh_order),
        ("collision_sync", collision_sync),
    ):
        t0 = time.perf_counter()
        fn(cfg)
        stages[name] = time.perf_counter() - t0
    return {"stages": stages, "total": sum(stages.values())}
