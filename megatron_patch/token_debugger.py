# Copyright (c) 2026
"""Token-level anomaly diagnostics for Megatron-LM."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import torch


class TokenAnomalyDebugger:
    """Collects high-frequency token IDs from anomaly steps only."""

    def __init__(self, enabled: bool = False, top_k: int = 10, preview_len: int = 64):
        self.enabled = enabled
        self.top_k = top_k
        self.preview_len = preview_len
        self.token_counter: Counter = Counter()

    def analyze(self, input_ids: Optional[torch.Tensor]) -> Optional[Dict[str, object]]:
        if not self.enabled or input_ids is None:
            return None

        if not torch.is_tensor(input_ids):
            return None

        flat = input_ids.detach().view(-1)
        if flat.numel() == 0:
            return None

        preview = flat[: self.preview_len].tolist()
        self.token_counter.update(preview)

        top_tokens = [
            {"token_id": int(token_id), "count": int(count)}
            for token_id, count in self.token_counter.most_common(self.top_k)
        ]
        return {
            "top_tokens": top_tokens,
            "input_token_preview": [int(x) for x in preview],
        }
