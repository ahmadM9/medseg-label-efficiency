import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Compose, EnsureChannelFirstd, LoadImaged

# in-context models read the task from labelled examples shown next to the
# query at test time; no weight is updated. each entry records where the
# weights come from and the training-data sentence the paper must carry.
IN_CONTEXT_MODELS = {
    "universeg": {
        "kind": "universeg",
        "source": "https://github.com/JJGO/UniverSeg",
        "input_size": 128,
        "training_data": "released weights trained on MegaMedical, which includes CAMUS "
        "under a different subject split; not a clean held-out result",
    },
    "seggpt": {
        "kind": "transformers",
        "source": "BAAI/seggpt-vit-large",
        "input_size": 448,
        "training_data": "no ultrasound or cardiac data in its training",
    },
}


def gray_uint8(image: np.ndarray) -> np.ndarray:
    lo, hi = float(image.min()), float(image.max())
    scaled = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)
    return (scaled * 255).astype(np.uint8)


def to_square(array: np.ndarray, size: int, mode: str) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    out = F.interpolate(t, size=(size, size), mode=mode, **kwargs)[0, 0]
    return out.round().clamp(0, 255).to(torch.uint8).numpy()


def load_pool(samples: list[dict], size: int) -> tuple[np.ndarray, np.ndarray]:
    # the whole support pool lives in memory at the model's square size:
    # images bilinear, label maps nearest, both uint8
    keys = ["image", "label"]
    load = Compose([LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)])
    images = np.empty((len(samples), size, size), dtype=np.uint8)
    labels = np.empty((len(samples), size, size), dtype=np.uint8)
    for i, sample in enumerate(samples):
        data = load({"image": sample["image"], "label": sample["label"]})
        images[i] = to_square(gray_uint8(data["image"][0].numpy()), size, "bilinear")
        labels[i] = to_square(data["label"][0].numpy(), size, "nearest")
    return images, labels


def filter_pool(pool: list[dict], query: dict, match_keys: list[str]) -> np.ndarray:
    # supports share the query's acquisition metadata (e.g. view) and never
    # its patient; the second rule holds by construction of the splits and
    # is enforced here anyway
    keep = [
        i
        for i, s in enumerate(pool)
        if s["patient"] != query["patient"] and all(s[k] == query[k] for k in match_keys)
    ]
    return np.asarray(keep, dtype=np.int64)


def draw_supports(candidates: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    # without replacement: a duplicate example carries no information, and
    # asking for more than the pool holds is a protocol error, not a warning
    if shots > len(candidates):
        raise ValueError(f"{shots} supports requested but only {len(candidates)} candidates")
    return rng.choice(candidates, size=shots, replace=False)


class UniversegBackend:
    def __init__(self, device: str):
        from universeg import universeg

        self.device = torch.device(device)
        self.model = universeg(pretrained=True).to(self.device).eval()

    def predict_draw(
        self, query: np.ndarray, support_images: np.ndarray, support_masks: np.ndarray
    ) -> np.ndarray:
        q = torch.from_numpy(query).float().div_(255)[None, None]
        s = torch.from_numpy(support_images).float().div_(255)[None, :, None]
        m = torch.from_numpy(support_masks).float()[None, :, None]
        with torch.no_grad():
            logits = self.model(q.to(self.device), s.to(self.device), m.to(self.device))
        return torch.sigmoid(logits)[0, 0].float().cpu().numpy()

    @staticmethod
    def combine(draws: list[np.ndarray]) -> np.ndarray:
        # the paper's recipe: mean of the per-draw probabilities, then 0.5
        return np.mean(draws, axis=0) > 0.5


class SeggptBackend:
    def __init__(self, device: str):
        from transformers import SegGptForImageSegmentation, SegGptImageProcessor

        source = IN_CONTEXT_MODELS["seggpt"]["source"]
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.processor = SegGptImageProcessor.from_pretrained(source)
        self.model = SegGptForImageSegmentation.from_pretrained(source, dtype=dtype)
        self.model.to(self.device).eval()

    def predict_draw(
        self, query: np.ndarray, support_images: np.ndarray, support_masks: np.ndarray
    ) -> np.ndarray:
        shots = len(support_images)
        rgb = [np.stack([g] * 3, axis=-1) for g in (query, *support_images)]
        # num_labels=1 paints the structure white on black; the post-processor
        # assigns each pixel to the nearer of the two, which is the model's
        # own decoding rule
        inputs = self.processor(
            images=[rgb[0]] * shots,
            prompt_images=rgb[1:],
            prompt_masks=[m.astype(np.uint8) for m in support_masks],
            num_labels=1,
            return_tensors="pt",
        )
        # the model casts the two image tensors to its dtype but not the
        # masks, which fails in fp16; cast everything floating here
        dtype = self.model.dtype
        inputs = {
            k: v.to(self.device, dtype) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out = self.model(**inputs, feature_ensemble=True, embedding_type="semantic")
        maps = self.processor.post_process_semantic_segmentation(out, num_labels=1)
        # feature ensemble averages the query features over the prompts, so
        # the k maps agree; the first one is the prediction
        return maps[0].cpu().numpy().astype(np.uint8)

    @staticmethod
    def combine(draws: list[np.ndarray]) -> np.ndarray:
        # majority vote over draws; a tie goes to background
        return np.sum(draws, axis=0) * 2 > len(draws)


def build_backend(model: str, device: str):
    if model not in IN_CONTEXT_MODELS:
        known = ", ".join(sorted(IN_CONTEXT_MODELS))
        raise KeyError(f"unknown in-context model {model!r}; registered: {known}")
    kind = IN_CONTEXT_MODELS[model]["kind"]
    return UniversegBackend(device) if kind == "universeg" else SeggptBackend(device)
