"""Promptable-model registry and the sam2 backend behind every prompted arm.

PROMPTABLE_MODELS is the plug-and-play point: each entry names a checkpoint,
the hydra config that builds its architecture, and (when the config is not
shipped inside the sam2 package) a vendored yaml to inject. Adding a model
variant — e.g. sam2_large for the scale ablation — is one new entry here.
"""

from pathlib import Path

import numpy as np

PROMPTABLE_MODELS = {
    "sam2": {
        "checkpoint": "checkpoints/sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "vendored_config": None,
    },
    "medsam2": {
        "checkpoint": "checkpoints/MedSAM2_latest.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t512.yaml",
        # MedSAM2 was fine-tuned at image size 512 with a config the sam2
        # package does not ship; download_checkpoints.sh fetches it and we
        # drop it into the installed package's config tree so hydra finds it.
        "vendored_config": "checkpoints/sam2.1_hiera_t512.yaml",
    },
}


class Sam2Backend:
    """Thin wrapper so tests can swap in a mock. predict() takes an RGB uint8
    image plus one prompt and returns one boolean mask."""

    def __init__(self, model: str, device: str):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if model not in PROMPTABLE_MODELS:
            known = ", ".join(sorted(PROMPTABLE_MODELS))
            raise KeyError(f"unknown promptable model {model!r}; registered: {known}")
        entry = PROMPTABLE_MODELS[model]
        self._ensure_vendored_config(entry)
        sam_model = build_sam2(entry["config"], entry["checkpoint"], device=torch.device(device))
        self.predictor = SAM2ImagePredictor(sam_model)
        # SAM2ImagePredictor hardcodes backbone feature sizes for 1024-pixel
        # input; models trained at other sizes need the pyramid levels scaled
        size = sam_model.image_size
        if size != 1024:
            self.predictor._bb_feat_sizes = [
                (size // 4, size // 4),
                (size // 8, size // 8),
                (size // 16, size // 16),
            ]

    @staticmethod
    def _ensure_vendored_config(entry: dict) -> None:
        if not entry["vendored_config"]:
            return
        import sam2 as sam2_pkg

        target = Path(sam2_pkg.__file__).parent / entry["config"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(Path(entry["vendored_config"]).read_text())

    def predict(self, image_rgb, box=None, point=None):
        self.predictor.set_image(image_rgb)
        kwargs = {}
        if box is not None:  # (r0, c0, r1, c1) -> sam's (x0, y0, x1, y1)
            r0, c0, r1, c1 = box
            kwargs["box"] = np.array([c0, r0, c1, r1])
        if point is not None:  # (row, col) -> sam's (x, y)
            kwargs["point_coords"] = np.array([[point[1], point[0]]])
            kwargs["point_labels"] = np.array([1])
        masks, scores, _ = self.predictor.predict(**kwargs, multimask_output=True)
        return masks[int(np.argmax(scores))].astype(bool)
