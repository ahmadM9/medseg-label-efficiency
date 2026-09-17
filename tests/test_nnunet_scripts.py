"""The nnU-Net converter and exporter on a synthetic CAMUS tree (no nnunetv2 needed)."""

import json
import sys

import nibabel as nib
import numpy as np
import pytest

from export_nnunet import main as export_main
from prepare_nnunet import BUDGET_OF_FOLD, DATASET_NAME, FOLDS, case_id, parse_case_id
from prepare_nnunet import main as prepare_main

# six train, two val, two test patients; subsets nested like the real lists
TRAIN = [f"patient{i:04d}" for i in range(1, 7)]
VAL = ["patient0007", "patient0008"]
TEST = ["patient0009", "patient0010"]
SUBSETS = {"train_p05.txt": TRAIN[:1], "train_p10.txt": TRAIN[:2], "train_p25.txt": TRAIN[:4]}


def label_map(shift: int) -> np.ndarray:
    label = np.zeros((60, 50), dtype=np.float32)  # CAMUS ships float32 labels
    label[10 + shift : 30 + shift, 10:30] = 2
    label[14 + shift : 30 + shift, 14:26] = 1
    label[35 + shift : 50 + shift, 15:35] = 3
    return label


@pytest.fixture
def camus_tree(tmp_path):
    root = tmp_path / "camus"
    (root / "database_split").mkdir(parents=True)
    for name, ids in (("training", TRAIN), ("validation", VAL), ("testing", TEST)):
        (root / "database_split" / f"subgroup_{name}.txt").write_text("\n".join(ids) + "\n")
    rng = np.random.default_rng(0)
    affine = np.diag([-0.3, -0.3, 1.0, 1.0])
    for n, patient in enumerate(TRAIN + VAL + TEST):
        pdir = root / "database_nifti" / patient
        pdir.mkdir(parents=True)
        for view in ("2CH", "4CH"):
            (pdir / f"Info_{view}.cfg").write_text("ImageQuality: Good\nEF: 55.0\n")
            for k, phase in enumerate(("ED", "ES")):
                image = rng.random((60, 50)).astype(np.float32) * 200
                nib.save(nib.Nifti1Image(image, affine), pdir / f"{patient}_{view}_{phase}.nii.gz")
                nib.save(nib.Nifti1Image(label_map(n % 3 + k), affine),
                         pdir / f"{patient}_{view}_{phase}_gt.nii.gz")
    subsets = tmp_path / "subsets"
    subsets.mkdir()
    for name, ids in SUBSETS.items():
        (subsets / name).write_text("\n".join(ids) + "\n")
    return root, subsets


def run_prepare(root, subsets, raw, limit=0):
    argv = ["prepare_nnunet.py", "--data-root", str(root), "--raw-dir", str(raw),
            "--subsets-dir", str(subsets)]
    if limit:
        argv += ["--limit", str(limit)]
    saved, sys.argv = sys.argv, argv
    try:
        prepare_main()
    finally:
        sys.argv = saved
    return raw / DATASET_NAME


def test_case_id_round_trip():
    sample = {"patient": "patient0042", "view": "4CH", "phase": "ES"}
    assert case_id(sample) == "CAMUS_patient0042_4CH_ES"
    assert parse_case_id(case_id(sample)) == sample
    with pytest.raises(ValueError):
        parse_case_id("CAMUS_patient0042_3CH_ES")


def test_prepare_layout_folds_and_label_dtype(camus_tree, tmp_path):
    root, subsets = camus_tree
    ds = run_prepare(root, subsets, tmp_path / "raw")
    images = sorted(p.name for p in (ds / "imagesTr").iterdir())
    labels = sorted(p.name for p in (ds / "labelsTr").iterdir())
    assert len(images) == len(labels) == (len(TRAIN) + len(VAL)) * 4
    assert len(list((ds / "imagesTs").iterdir())) == len(TEST) * 4
    assert images[0] == "CAMUS_patient0001_2CH_ED_0000.nii.gz"
    assert labels[0] == "CAMUS_patient0001_2CH_ED.nii.gz"
    lbl = nib.load(ds / "labelsTr" / labels[0])
    assert lbl.get_data_dtype() == np.uint8
    assert set(np.unique(np.asanyarray(lbl.dataobj))) == {0, 1, 2, 3}
    # the affine survives the re-save, so predictions come back on the native grid
    src = nib.load(root / "database_nifti/patient0001/patient0001_2CH_ED_gt.nii.gz")
    assert np.allclose(lbl.affine, src.affine)

    dataset = json.loads((ds / "dataset.json").read_text())
    assert dataset["labels"] == {"background": 0, "lv_endo": 1, "lv_myo": 2, "left_atrium": 3}
    assert dataset["numTraining"] == (len(TRAIN) + len(VAL)) * 4
    assert dataset["file_ending"] == ".nii.gz"

    splits = json.loads((ds / "splits_final.json").read_text())
    assert len(splits) == len(FOLDS) == 4
    expected = {0: 1, 1: 2, 2: 4, 3: len(TRAIN)}
    for fold, split in enumerate(splits):
        assert len(split["train"]) == expected[fold] * 4, BUDGET_OF_FOLD[fold]
        assert split["val"] == splits[0]["val"] and len(split["val"]) == len(VAL) * 4
        patients = {parse_case_id(c)["patient"] for c in split["train"] + split["val"]}
        assert not patients & set(TEST)
        assert not set(parse_case_id(c)["patient"] for c in split["train"]) & set(VAL)
    # budgets nest like the subset lists
    for fold in range(3):
        assert set(splits[fold]["train"]) <= set(splits[fold + 1]["train"])


def test_prepare_limit_keeps_every_fold_non_empty(camus_tree, tmp_path):
    root, subsets = camus_tree
    ds = run_prepare(root, subsets, tmp_path / "raw", limit=1)
    splits = json.loads((ds / "splits_final.json").read_text())
    assert [len(s["train"]) for s in splits] == [4, 4, 4, 4]
    assert all(len(s["val"]) == 4 for s in splits)
    assert len(list((ds / "imagesTs").iterdir())) == 4


def test_export_scores_predictions_and_writes_masks(camus_tree, tmp_path, monkeypatch, capsys):
    root, subsets = camus_tree
    ds = run_prepare(root, subsets, tmp_path / "raw")
    # predictions equal to the ground truth except one structure dropped on one image
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    for patient in TEST:
        for view in ("2CH", "4CH"):
            for phase in ("ED", "ES"):
                gt = nib.load(root / f"database_nifti/{patient}/{patient}_{view}_{phase}_gt.nii.gz")
                data = np.rint(np.asanyarray(gt.dataobj)).astype(np.uint8)
                if patient == TEST[0] and view == "2CH" and phase == "ED":
                    data[data == 3] = 0
                case = case_id({"patient": patient, "view": view, "phase": phase})
                nib.save(nib.Nifti1Image(data, gt.affine), pred_dir / f"{case}.nii.gz")
    out_dir = tmp_path / "outputs/nnunet_full"
    monkeypatch.setattr(sys, "argv", ["export_nnunet.py", "--pred-dir", str(pred_dir),
                                      "--out-dir", str(out_dir), "--data-root", str(root),
                                      "--fold", "3", "--no-mirroring"])
    export_main()
    summary = json.loads((out_dir / "test_metrics.json").read_text())
    assert summary["n_samples"] == len(TEST) * 4
    assert summary["dice_lv_endo"] == 1.0 and summary["dice_lv_myo"] == 1.0
    assert summary["dice_left_atrium"] == pytest.approx(7 / 8)
    assert summary["fold"] == 3 and summary["mirroring"] is False and summary["postprocessing"]
    assert "without mirroring" in summary["protocol"]
    rows = (out_dir / "test_per_sample.csv").read_text().splitlines()
    assert rows[0].startswith("patient,view,phase,quality,dice_lv_endo,hd95_lv_endo")
    assert len(rows) == 1 + len(TEST) * 4
    assert len(list((out_dir / "masks").glob("*.npz"))) == len(TEST) * 4
    with np.load(out_dir / "masks" / f"{TEST[1]}_4CH_ES.npz") as m:
        assert m["mask"].dtype == np.uint8 and m["mask"].shape == (60, 50)
    assert ds.exists()


def test_export_rejects_wrong_shape_and_missing_prediction(camus_tree, tmp_path, monkeypatch):
    root, _ = camus_tree
    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    case = case_id({"patient": TEST[0], "view": "2CH", "phase": "ED"})
    nib.save(nib.Nifti1Image(np.zeros((30, 25), dtype=np.uint8), np.eye(4)),
             pred_dir / f"{case}.nii.gz")
    argv = ["export_nnunet.py", "--pred-dir", str(pred_dir), "--out-dir", str(tmp_path / "o"),
            "--data-root", str(root), "--limit", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="shape"):
        export_main()
    monkeypatch.setattr(sys, "argv", argv[:-2] + ["--limit", "2"])
    (pred_dir / f"{case}.nii.gz").unlink()
    with pytest.raises(SystemExit, match="no prediction"):
        export_main()
