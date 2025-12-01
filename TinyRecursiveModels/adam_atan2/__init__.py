from __future__ import annotations

import warnings
from typing import Any, Iterable, Tuple, Union

import torch
from torch.optim import AdamW


class AdamATan2(AdamW):
    """Fallback optimizer using AdamW when fused adam_atan2 extension is unavailable."""

    def __init__(
        self,
        params: Iterable[Any],
        lr: Union[float, torch.Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "adam_atan2 CUDA extension not available; falling back to AdamW."
            " Training dynamics may differ from the original implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        super().__init__(params, lr=lr, betas=betas, weight_decay=weight_decay, **kwargs)
