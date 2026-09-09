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

**2026-09-09 — Kaggle's default GPU (P100) is unsupported by current PyTorch.**
Training died with `CUDA error: no kernel image is available for execution on the
device`: kernels get a Tesla P100 (Pascal, sm_60) by default, and recent PyTorch
wheels no longer ship Pascal kernels. Fix: request a T4 explicitly, which needs a
current kaggle CLI (2.x): `kaggle kernels push -p kaggle/ --accelerator NvidiaTeslaT4`.
