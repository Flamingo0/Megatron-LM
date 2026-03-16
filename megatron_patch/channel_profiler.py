# Copyright (c) 2026
"""Channel-level training stability profiler for Megatron-LM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ChannelStats:
    """Running statistics for one data channel."""

    steps: int = 0
    loss_sum: float = 0.0
    grad_sum: float = 0.0
    anomaly_count: int = 0

    def update(self, loss: Optional[float], grad_norm: Optional[float], is_anomaly: bool) -> None:
        self.steps += 1
        if loss is not None:
            self.loss_sum += float(loss)
        if grad_norm is not None:
            self.grad_sum += float(grad_norm)
        if is_anomaly:
            self.anomaly_count += 1

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / self.steps if self.steps else 0.0

    @property
    def avg_grad_norm(self) -> float:
        return self.grad_sum / self.steps if self.steps else 0.0

    @property
    def anomaly_rate(self) -> float:
        return self.anomaly_count / self.steps if self.steps else 0.0


class ChannelProfiler:
    """Tracks per-channel loss/grad behavior and anomaly rates."""

    def __init__(self, enabled: bool = True, report_interval: int = 1000):
        self.enabled = enabled
        self.report_interval = max(1, report_interval)
        self.channel_stats: Dict[str, ChannelStats] = {}

    def update(
        self,
        channel: str,
        loss: Optional[float],
        grad_norm: Optional[float],
        is_anomaly: bool,
        step: int,
    ) -> Optional[Dict[str, Dict[str, float]]]:
        if not self.enabled:
            return None

        name = channel if channel else "unknown"
        if name not in self.channel_stats:
            self.channel_stats[name] = ChannelStats()

        self.channel_stats[name].update(loss=loss, grad_norm=grad_norm, is_anomaly=is_anomaly)

        if step % self.report_interval == 0:
            return self.snapshot()
        return None

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Returns compact aggregate metrics for all channels."""
        output: Dict[str, Dict[str, float]] = {}
        for channel, stats in self.channel_stats.items():
            output[channel] = {
                "steps": float(stats.steps),
                "avg_loss": stats.avg_loss,
                "avg_grad_norm": stats.avg_grad_norm,
                "anomaly_count": float(stats.anomaly_count),
                "anomaly_rate": stats.anomaly_rate,
            }
        return output
