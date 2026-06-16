import random
import numpy as np
import torch


def set_seed(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("WARN: CUDA not available, falling back to CPU")
            return torch.device("cpu")
        if ":" in device_str:
            try:
                idx = int(device_str.split(":")[1])
            except Exception:
                print(f"WARN: Failed to parse device '{device_str}', using cuda:0")
                return torch.device("cuda:0")
            if idx >= torch.cuda.device_count():
                print(f"WARN: {device_str} exceeds available GPUs ({torch.cuda.device_count()}), using cuda:0")
                return torch.device("cuda:0")
        return torch.device(device_str)
    return torch.device("cpu")


def normalize_iq(x_np: np.ndarray) -> np.ndarray:
    x = x_np.astype(np.float32, copy=False)
    assert x.ndim == 2 and x.shape[0] == 2, f"Expected (2, L), got {x.shape}"
    L = x.shape[1]
    power = np.sum(x[0] ** 2 + x[1] ** 2) / float(L)
    scale = np.sqrt(max(power, 1e-12))
    x[0] /= scale
    x[1] /= scale
    return x


def energy_from_logits(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return -T * torch.logsumexp(logits / T, dim=1)