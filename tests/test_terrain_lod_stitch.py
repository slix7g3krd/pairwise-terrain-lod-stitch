import pytest
import graphics.terrain_lod_stitch as mod

CASES = [
    ("stitch_border", (1)),
    ("mip_screen_error", (1)),
    ("mesh_order", (1)),
    ("unload_refcount", (1)),
    ("lod_determinism", (1)),
    ("normal_rebuild", (1)),
    ("height_sample", (1)),
    ("collision_sync", (1)),
    ("validation", (1)),
    ("audit", (1)),
    ("fallback", (1)),
    ("metrics", (1)),
    ("stream_budget", (1)),
    ("reenter", (1)),
    ("profile", (1)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
