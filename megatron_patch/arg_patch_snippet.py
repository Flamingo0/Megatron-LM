# Copyright (c) 2026
"""Helper snippet for external argument-patch scripts.

This module is intended for users who maintain their own `get_patch_args(parser)`
style script and want to keep anomaly-monitor options in sync with Megatron.
"""

from __future__ import annotations

import argparse
from typing import Callable, Union


PatchTarget = Union[argparse._ArgumentGroup, argparse.ArgumentParser]
PatchFn = Callable[..., object]


def _patch_flag_if_not_exist(group_or_parser: PatchTarget, keyname: str, help_text: str) -> None:
    has_keyname = False
    for action in vars(group_or_parser)["_actions"]:
        if keyname in action.option_strings:
            has_keyname = True
            break

    if not has_keyname:
        group_or_parser.add_argument(keyname, action="store_true", help=help_text)


def add_anomaly_monitor_patch_args(group: PatchTarget, patch_if_not_exist: PatchFn) -> None:
    """Add anomaly-monitor related arguments idempotently.

    Example usage inside custom `get_patch_args(parser)`:

    ```python
    group = parser.add_argument_group(title="patch")
    add_anomaly_monitor_patch_args(group, patch_if_not_exist)
    ```
    """

    _patch_flag_if_not_exist(
        group,
        "--enable-anomaly-monitor",
        "Enable low-intrusion anomaly monitoring for training loop.",
    )
    patch_if_not_exist(
        group,
        "--anomaly-start-step",
        type=int,
        default=100,
        help="Global step to start anomaly monitor warmup window collection.",
    )
    patch_if_not_exist(
        group,
        "--anomaly-window-size",
        type=int,
        default=100,
        help="Sliding-window size used for moving average baselines.",
    )
    patch_if_not_exist(
        group,
        "--anomaly-loss-multiplier",
        type=float,
        default=3.0,
        help="Relative spike threshold for loss: moving_avg_loss * multiplier.",
    )
    patch_if_not_exist(
        group,
        "--anomaly-grad-multiplier",
        type=float,
        default=3.0,
        help="Relative spike threshold for grad_norm: moving_avg_grad * multiplier.",
    )
    patch_if_not_exist(
        group,
        "--anomaly-loss-abs-max",
        type=float,
        default=1000.0,
        help="Absolute loss threshold for anomaly detection.",
    )
    patch_if_not_exist(
        group,
        "--anomaly-grad-abs-max",
        type=float,
        default=100000.0,
        help="Absolute grad_norm threshold for anomaly detection.",
    )
    patch_if_not_exist(
        group,
        "--anomaly-cooldown-steps",
        type=int,
        default=50,
        help="Suppress repeated anomaly logs within this step interval.",
    )
    patch_if_not_exist(
        group,
        "--anomaly-buffer-size",
        type=int,
        default=128,
        help="Buffered anomaly log entries before flushing JSONL to disk.",
    )
    patch_if_not_exist(
        group,
        "--anomaly-flush-interval",
        type=int,
        default=100,
        help="Periodic flush interval in global train steps.",
    )
    patch_if_not_exist(
        group,
        "--anomaly-output-file",
        type=str,
        default="anomaly_log.jsonl",
        help="JSONL output path for anomaly records.",
    )
    _patch_flag_if_not_exist(
        group,
        "--enable-channel-profiler",
        "Enable per-channel anomaly/quality profiling.",
    )
    _patch_flag_if_not_exist(
        group,
        "--enable-token-debugger",
        "Enable anomaly-step token frequency diagnostics.",
    )
