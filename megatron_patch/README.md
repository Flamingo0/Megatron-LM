# Megatron Patch: Low-intrusion Anomaly Monitoring

## 1) System architecture

`AnomalyMonitor` is the coordinator and is called once per train step:

```python
monitor.observe(step, loss, grad_norm, lr, batch, found_inf_flag=found_inf)
```

Internally it executes three paths:

1. **Anomaly detection**: NaN/Inf, relative spike (`moving_avg * multiplier`), absolute max, FP16 overflow.
2. **Channel profiling**: running per-channel averages and anomaly counts.
3. **Token debugging**: anomaly-step-only top-token statistics and token preview.

## 2) Training loop patch example (minimal intrusion)

```python
# 1) build monitor once after args are parsed
from megatron_patch import AnomalyMonitor, MonitorConfig

monitor = AnomalyMonitor(
    MonitorConfig(
        enabled=args.enable_anomaly_monitor,
        monitor_start_step=args.anomaly_start_step,
        window_size=args.anomaly_window_size,
        loss_multiplier=args.anomaly_loss_multiplier,
        grad_multiplier=args.anomaly_grad_multiplier,
        loss_absolute_max=args.anomaly_loss_abs_max,
        grad_absolute_max=args.anomaly_grad_abs_max,
        cooldown_steps=args.anomaly_cooldown_steps,
        buffer_size=args.anomaly_buffer_size,
        flush_interval=args.anomaly_flush_interval,
        output_file=args.anomaly_output_file,
        enable_channel_profiler=args.enable_channel_profiler,
        enable_token_debugger=args.enable_token_debugger,
    )
)

# 2) call observe in train loop after train_step
found_inf = bool(skipped_iter) and (loss_scale < previous_loss_scale)
monitor.observe(
    step=iteration,
    loss=loss_dict.get("lm loss") if isinstance(loss_dict, dict) else None,
    grad_norm=grad_norm,
    lr=learning_rate,
    batch=batch,  # expected dataloader schema in design doc
    found_inf_flag=found_inf,
    dp_rank=mpu.get_data_parallel_rank(with_context_parallel=True),
    tp_rank=mpu.get_tensor_model_parallel_rank(),
    pp_rank=mpu.get_pipeline_model_parallel_rank(),
    pp_world_size=mpu.get_pipeline_model_parallel_world_size(),
    dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
)

# 3) flush on exit/checkpoint boundaries
monitor.flush()
```

## 3) Distributed-safe behavior

Only one rank writes files to avoid duplicated logs:

- `dp_rank == 0`
- `tp_rank == 0`
- `pp_rank == pp_world_size - 1`

Loss is optionally reduced with DP mean before anomaly checks.

## 4) Sliding-window behavior

Warmup-safe detection:

1. `step < monitor_start_step`: collect window only, no anomaly check.
2. `monitor_start_step <= step < monitor_start_step + window_size`: continue building moving averages.
3. `step >= monitor_start_step + window_size`: enable anomaly checks.

## 5) IO buffer behavior

Entries are staged in-memory and flushed to JSONL when:

- `len(buffer) >= buffer_size`, or
- `step % flush_interval == 0`, or
- explicit `monitor.flush()/monitor.close()`.

## 6) JSONL log example

```json
{"step": 12031, "loss": 6.31, "grad_norm": 1204.8, "lr": 2e-5, "dataset": "pile", "channel": "code", "sample_ids": [112, 113], "found_inf_flag": false, "moving_avg_loss": 2.44, "moving_avg_grad": 212.3, "reasons": ["loss_spike_relative", "grad_spike_relative"], "top_tokens": [{"token_id": 198, "count": 9}], "input_token_preview": [1, 198, 345, 76]}
```

## 7) CLI arguments to add in Megatron arguments parser

```python
group.add_argument('--enable-anomaly-monitor', action='store_true')
group.add_argument('--anomaly-start-step', type=int, default=100)
group.add_argument('--anomaly-window-size', type=int, default=100)
group.add_argument('--anomaly-loss-multiplier', type=float, default=3.0)
group.add_argument('--anomaly-grad-multiplier', type=float, default=3.0)
group.add_argument('--anomaly-loss-abs-max', type=float, default=1000.0)
group.add_argument('--anomaly-grad-abs-max', type=float, default=100000.0)
group.add_argument('--anomaly-cooldown-steps', type=int, default=50)
group.add_argument('--anomaly-buffer-size', type=int, default=128)
group.add_argument('--anomaly-flush-interval', type=int, default=100)
group.add_argument('--anomaly-output-file', type=str, default='anomaly_log.jsonl')
group.add_argument('--enable-channel-profiler', action='store_true')
group.add_argument('--enable-token-debugger', action='store_true')
```
