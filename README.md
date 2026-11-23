# pairwise-terrain-lod-stitch

## 领域

游戏、图形与多媒体

## 主题

地形 LOD 缝合

## 说明

块边界缝合、mip 按屏幕误差与卸载引用计数。

初始状态下 `graphics/terrain_lod_stitch.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
