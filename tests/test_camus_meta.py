"""Tests for CAMUS metadata handling that run without the dataset."""

from medseg_label_efficiency.data.camus import LABELS, find_nii, parse_info


def test_find_nii_prefers_gz_but_accepts_plain(tmp_path):
    (tmp_path / "a.nii.gz").touch()
    (tmp_path / "b.nii").touch()
    assert find_nii(tmp_path, "a").name == "a.nii.gz"
    assert find_nii(tmp_path, "b").name == "b.nii"
    # missing files resolve to the canonical name so error messages stay clear
    assert find_nii(tmp_path, "c").name == "c.nii.gz"


def test_parse_info(tmp_path):
    cfg = tmp_path / "Info_2CH.cfg"
    cfg.write_text("ED: 1\nES: 18\nNbFrame: 18\nSex: F\nAge: 56\nImageQuality: Good\nEF: 54\n")
    info = parse_info(cfg)
    assert info["ImageQuality"] == "Good"
    assert info["EF"] == "54"
    assert info["ED"] == "1"


def test_parse_info_ignores_malformed_lines(tmp_path):
    cfg = tmp_path / "info.cfg"
    cfg.write_text("Quality: Poor\nnot a key value line\n")
    info = parse_info(cfg)
    assert info == {"Quality": "Poor"}


def test_label_map():
    assert set(LABELS) == {0, 1, 2, 3}
    assert LABELS[1] == "lv_endo"
