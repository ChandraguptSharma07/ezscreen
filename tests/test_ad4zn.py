import pytest

from ezscreen.benchmark.ad4zn import build_ad4zn_map_spec, validate_ad4zn_map_manifest


def _manifest(spec):
    result = spec.as_dict()
    result["tz_pseudoatoms"] = True
    result["map_files"] = {"ace.OA.map": "a" * 64}
    return result


def test_ad4zn_map_spec_covers_requested_box_with_even_grid_points():
    spec = build_ad4zn_map_spec([40.7317, 32.8822, 47.3635], [36.8194, 41.835, 43.7048])

    assert spec.npts == (100, 112, 118)
    assert all(
        actual >= requested
        for actual, requested in zip(spec.covered_size, spec.requested_size)
    )


def test_ad4zn_manifest_requires_tz_ad4zn_and_exact_box():
    spec = build_ad4zn_map_spec([1, 2, 3], [20, 20, 20])
    manifest = _manifest(spec)

    validate_ad4zn_map_manifest(manifest, spec)

    manifest["tz_pseudoatoms"] = False
    with pytest.raises(ValueError, match="TZ"):
        validate_ad4zn_map_manifest(manifest, spec)


def test_ad4zn_manifest_rejects_box_or_hash_mismatch():
    spec = build_ad4zn_map_spec([1, 2, 3], [20, 20, 20])
    manifest = _manifest(spec)
    manifest["npts"] = [52, 52, 54]
    with pytest.raises(ValueError, match="npts"):
        validate_ad4zn_map_manifest(manifest, spec)

    manifest = _manifest(spec)
    manifest["map_files"] = {"ace.OA.map": "bad"}
    with pytest.raises(ValueError, match="hash"):
        validate_ad4zn_map_manifest(manifest, spec)
