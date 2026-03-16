# Copyright (c) 2026
"""Low-intrusion training anomaly monitor for Megatron-LM."""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import torch
import torch.distributed as dist

from megatron_patch.channel_profiler import ChannelProfiler
from megatron_patch.token_debugger import TokenAnomalyDebugger


@dataclass
class MonitorConfig:
    enabled: bool = False
    monitor_start_step: int = 0
    window_size: int = 100
    loss_multiplier: float = 3.0
    grad_multiplier: float = 3.0
    loss_absolute_max: float = 1e3
    grad_absolute_max: float = 1e5
    cooldown_steps: int = 50
    buffer_size: int = 128
    flush_interval: int = 100
    output_file: str = "anomaly_log.jsonl"
    enable_channel_profiler: bool = True
    enable_token_debugger: bool = False
    channel_report_interval: int = 1000


class AnomalyMonitor:
    """Monitors loss/grad spikes with buffered JSONL logging."""

    def __init__(self, config: MonitorConfig):
        self.cfg = config
        self.loss_window: Deque[float] = deque(maxlen=max(1, config.window_size))
        self.grad_window: Deque[float] = deque(maxlen=max(1, config.window_size))

        self.last_anomaly_step = -10**9
        self.buffer: List[Dict[str, object]] = []

        self.channel_profiler = ChannelProfiler(
            enabled=config.enable_channel_profiler,
            report_interval=config.channel_report_interval,
        )
        self.token_debugger = TokenAnomalyDebugger(enabled=config.enable_token_debugger)

    def _safe_float(self, value: object) -> Optional[float]:
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 0:
                return None
            return float(value.detach().float().mean().item())
        return float(value)

    def _is_finite(self, value: Optional[float]) -> bool:
        return value is not None and math.isfinite(value)

    def _moving_avg(self, window: Deque[float]) -> Optional[float]:
        if not window:
            return None
        return sum(window) / len(window)

    def _extract_meta(self, batch: Optional[dict]) -> Dict[str, object]:
        meta = (batch or {}).get("meta", {})
        return {
            "dataset": meta.get("dataset", "unknown"),
            "channel": meta.get("channel", "unknown"),
            "sample_ids": list(meta.get("sample_ids", []))[:64],
        }

    def _reduce_value_dp_mean(self, value: Optional[float], group=None) -> Optional[float]:
        if value is None:
            return None
        if not dist.is_available() or not dist.is_initialized():
            return value
        tensor = torch.tensor(value, device="cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        world_size = dist.get_world_size(group=group)
        return float((tensor / world_size).item())

    def _should_log_on_this_rank(self, dp_rank: int, tp_rank: int, pp_rank: int, pp_world_size: int) -> bool:
        # Distributed-safe rule: dp=0, tp=0, pp=last-stage.
        return dp_rank == 0 and tp_rank == 0 and pp_rank == (pp_world_size - 1)

    def observe(
        self,
        step: int,
        loss: object,
        grad_norm: object,
        lr: object,
        batch: Optional[dict],
        found_inf_flag: bool = False,
        dp_rank: int = 0,
        tp_rank: int = 0,
        pp_rank: int = 0,
        pp_world_size: int = 1,
        dp_group=None,
    ) -> Optional[Dict[str, object]]:
        """Observe one training step.

        Returns anomaly entry dict when one is recorded, else None.
        """
        if not self.cfg.enabled:
            return None

        loss_v = self._reduce_value_dp_mean(self._safe_float(loss), group=dp_group)
        grad_v = self._safe_float(grad_norm)
        lr_v = self._safe_float(lr)

        if self._is_finite(loss_v):
            self.loss_window.append(loss_v)
        if self._is_finite(grad_v):
            self.grad_window.append(grad_v)

        if step < self.cfg.monitor_start_step + self.cfg.window_size:
            self._update_channel_profiler(step, loss_v, grad_v, False, batch)
            self._flush_if_needed(step)
            return None

        moving_avg_loss = self._moving_avg(self.loss_window)
        moving_avg_grad = self._moving_avg(self.grad_window)

        reasons = self._detect_reasons(loss_v, grad_v, moving_avg_loss, moving_avg_grad, found_inf_flag)
        is_anomaly = len(reasons) > 0
        self._update_channel_profiler(step, loss_v, grad_v, is_anomaly, batch)

        if not is_anomaly:
            self._flush_if_needed(step)
            return None

        if step - self.last_anomaly_step < self.cfg.cooldown_steps:
            self._flush_if_needed(step)
            return None

        self.last_anomaly_step = step

        if not self._should_log_on_this_rank(dp_rank, tp_rank, pp_rank, pp_world_size):
            return None

        meta = self._extract_meta(batch)
        token_diag = self.token_debugger.analyze((batch or {}).get("input_ids"))

        entry: Dict[str, object] = {
            "step": int(step),
            "loss": loss_v,
            "grad_norm": grad_v,
            "lr": lr_v,
            "dataset": meta["dataset"],
            "channel": meta["channel"],
            "sample_ids": meta["sample_ids"],
            "found_inf_flag": bool(found_inf_flag),
            "moving_avg_loss": moving_avg_loss,
            "moving_avg_grad": moving_avg_grad,
            "reasons": reasons,
        }
        if token_diag is not None:
            entry.update(token_diag)

        self.buffer.append(entry)
        self._flush_if_needed(step, force=len(self.buffer) >= self.cfg.buffer_size)
        return entry

    def _detect_reasons(
        self,
        loss_v: Optional[float],
        grad_v: Optional[float],
        moving_avg_loss: Optional[float],
        moving_avg_grad: Optional[float],
        found_inf_flag: bool,
    ) -> List[str]:
        reasons: List[str] = []

        if loss_v is None or math.isnan(loss_v):
            reasons.append("loss_nan")
        elif math.isinf(loss_v):
            reasons.append("loss_inf")

        if grad_v is None or math.isnan(grad_v):
            reasons.append("grad_norm_nan")
        elif math.isinf(grad_v):
            reasons.append("grad_norm_inf")

        if (
            self._is_finite(loss_v)
            and self._is_finite(moving_avg_loss)
            and moving_avg_loss > 0
            and loss_v > moving_avg_loss * self.cfg.loss_multiplier
        ):
            reasons.append("loss_spike_relative")

        if (
            self._is_finite(grad_v)
            and self._is_finite(moving_avg_grad)
            and moving_avg_grad > 0
            and grad_v > moving_avg_grad * self.cfg.grad_multiplier
        ):
            reasons.append("grad_spike_relative")

        if self._is_finite(loss_v) and loss_v > self.cfg.loss_absolute_max:
            reasons.append("loss_spike_absolute")
        if self._is_finite(grad_v) and grad_v > self.cfg.grad_absolute_max:
            reasons.append("grad_spike_absolute")

        if found_inf_flag:
            reasons.append("fp16_overflow_found_inf")

        return reasons

    def _update_channel_profiler(
        self,
        step: int,
        loss_v: Optional[float],
        grad_v: Optional[float],
        is_anomaly: bool,
        batch: Optional[dict],
    ) -> None:
        channel = self._extract_meta(batch).get("channel", "unknown")
        report = self.channel_profiler.update(
            channel=channel,
            loss=loss_v,
            grad_norm=grad_v,
            is_anomaly=is_anomaly,
            step=step,
        )
        if report is not None and self._should_print_rank0():
            print(f"[AnomalyMonitor] channel_profile@step={step}: {json.dumps(report, ensure_ascii=False)}")

    def _should_print_rank0(self) -> bool:
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _flush_if_needed(self, step: int, force: bool = False) -> None:
        if not self.buffer:
            return
        if not force and (step % self.cfg.flush_interval != 0):
            return
        self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        os.makedirs(os.path.dirname(self.cfg.output_file) or ".", exist_ok=True)
        with open(self.cfg.output_file, "a", encoding="utf-8") as f:
            for item in self.buffer:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        self.buffer.clear()

    def state_dict(self) -> Dict[str, object]:
        return {
            "last_anomaly_step": self.last_anomaly_step,
            "loss_window": list(self.loss_window),
            "grad_window": list(self.grad_window),
            "channel_stats": self.channel_profiler.snapshot(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.last_anomaly_step = int(state.get("last_anomaly_step", self.last_anomaly_step))
        self.loss_window.clear()
        self.grad_window.clear()
        for value in state.get("loss_window", []):
            self.loss_window.append(float(value))
        for value in state.get("grad_window", []):
            self.grad_window.append(float(value))

    def close(self) -> None:
        self.flush()
