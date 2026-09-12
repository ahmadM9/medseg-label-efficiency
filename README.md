# medseg-label-efficiency

How many labeled patients does each way of using a segmentation model need
before it beats the others? Train from scratch, adapt a foundation model, or
show it examples at test time.

This repo compares these strategies on the [CAMUS echocardiography dataset](https://www.creatis.insa-lyon.fr/Challenge/camus/)
(LV endocardium, LV myocardium, left atrium) at label budgets of 20, 40, 100
and 400 training patients:

- a plain 2D U-Net ([MONAI](https://monai.io/)) trained supervised, automatic
- [SAM 2.1](https://github.com/facebookresearch/sam2) and [MedSAM2](https://github.com/bowang-lab/MedSAM2) with simulated prompts, zero-shot
- MedSAM2 with its mask decoder fine-tuned on the budget, prompted
- frozen [DINOv2](https://github.com/facebookresearch/dinov2) and [DINOv3](https://github.com/facebookresearch/dinov3) features with a small trained head, automatic
- [UniverSeg](https://github.com/JJGO/UniverSeg) in-context, no training: its released weights were trained on MegaMedical, which includes CAMUS under a different subject split, so its scores are not a clean held-out result (weights under the OpenRAIL++-M licence, research use only)
- [SegGPT](https://github.com/baaivision/Painter) in-context, no training: no ultrasound or cardiac data in its training

**Status: work in progress.** Results land here as they are produced.

## Data setup

The CAMUS dataset requires (free) registration:

1. Create an account at the [Human Heart Project](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8)
   and download the NIfTI release (`database_nifti` + `database_split`, ~3.6 GB).
2. Link or point the repo at it:

   ```bash
   ./scripts/link_data.sh /path/to/camus   # creates the data/camus symlink
   python -m medseg_label_efficiency.data.verify --data-root data/camus
   ```

The verify step checks that all 500 patients and the official
train/val/test split files (400/50/50) are present and consistent.

Data is never committed to this repo.

## Development

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest          # tests that need the dataset auto-skip if it is not linked
ruff check .
```

## Citations

This project uses the CAMUS dataset, which requires citing:

> S. Leclerc, E. Smistad, J. Pedrosa, A. Østvik, et al.
> "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D
> Echocardiography," IEEE Transactions on Medical Imaging, vol. 38, no. 9,
> pp. 2198-2210, 2019. doi:10.1109/TMI.2019.2900516

## License

Apache-2.0, see [LICENSE](LICENSE). The CAMUS data itself is governed by its
own [license terms](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8).
