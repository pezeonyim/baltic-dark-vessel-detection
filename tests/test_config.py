import pytest

from maritime.config import BBox, available_regions, load_region


def test_all_region_configs_load():
    names = available_regions()
    assert "baltic" in names
    for name in names:
        r = load_region(name)
        assert r.name == name
        assert r.default_start <= r.default_end


def test_bbox_rejects_inverted_ranges():
    with pytest.raises(ValueError):
        BBox(min_lon=10, min_lat=0, max_lon=5, max_lat=1)


def test_bbox_contains():
    b = load_region("baltic").bbox
    assert b.contains(20.0, 58.0)        # central Baltic
    assert not b.contains(-70.0, 40.0)   # off New England
