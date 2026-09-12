# Friction log

Notes on anything that was harder than it should have been while building this
project — unclear docs, surprising defaults, bugs in dependencies. Kept so that
fixable upstream issues (MONAI, SAM 2, MedSAM2) turn into issues/PRs instead of
being forgotten.

Format: date, what happened, where, whether it's worth reporting upstream.

---

**2026-09-09 — Kaggle silently gunzips `.nii.gz` files in dataset uploads.**
Uploaded the CAMUS NIfTI release as a private dataset; on the kernel mount every
`.nii.gz` had become a plain `.nii` (Kaggle decompresses archives during dataset
processing, including gzip members inside a zip). Anything that builds file paths
with a hardcoded `.nii.gz` extension breaks. Fixed on our side by resolving either
extension in the loader and verifier. Platform behavior, not a library bug — noted
here because it will bite anyone hosting medical imaging data on Kaggle.

**2026-09-09 — SAM2ImagePredictor breaks with non-1024 configs (MedSAM2's 512).**
`SAM2ImagePredictor` hardcodes `_bb_feat_sizes` for 1024-pixel input; loading
MedSAM2 (fine-tuned at image size 512) through it fails in `set_image` with a
tensor view/stride RuntimeError. Workaround: override `_bb_feat_sizes` with
sizes derived from `model.image_size` (size/4, size/8, size/16). MedSAM2's own
inference scripts sidestep this, but anyone loading their checkpoint through
the standard sam2 predictor hits it. Worth an upstream issue on sam2 (derive
the sizes from the model instead of hardcoding) — candidate contribution.

**2026-09-09 — MedSAM2's model config is not in the sam2 package.**
The checkpoint needs the repo's `sam2/configs/sam2.1_hiera_t512.yaml`, which
pip-installed sam2 doesn't ship; hydra only searches the package config tree.
Our download script fetches the yaml and the harness copies it into the
installed package's config directory at load time.

**2026-09-09 — Kaggle's default GPU (P100) is unsupported by current PyTorch.**
Training died with `CUDA error: no kernel image is available for execution on the
device`: kernels get a Tesla P100 (Pascal, sm_60) by default, and recent PyTorch
wheels no longer ship Pascal kernels. Fix: request a T4 explicitly, which needs a
current kaggle CLI (2.x): `kaggle kernels push -p kaggle/ --accelerator NvidiaTeslaT4`.

**2026-09-12 - transformers SegGPT casts images but not masks to the model dtype.**
`SegGptModel.forward` converts `pixel_values` and `prompt_pixel_values` to the
weight dtype and then feeds `prompt_masks` (still float32) through the same
patch embedding, so a model loaded in fp16 fails with `Input type (float) and
bias type (c10::Half) should be the same` (transformers 5.x). Fixed on our side
by casting every floating input to `model.dtype` before the call. Worth an
upstream one-line fix (cast `prompt_masks` alongside the other two), candidate
contribution.
