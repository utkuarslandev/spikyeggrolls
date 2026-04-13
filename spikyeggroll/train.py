"""Main training loop using EGGROLL evolution strategies."""

import argparse
from collections import defaultdict, deque
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import optax

from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.models.common import simple_es_tree_key

from spikyeggroll.configs import SNNConfig
from spikyeggroll.data.cifar10 import augment_batch as _augment_cifar
from spikyeggroll.runtime import get_dataset_spec, get_model_cls


def _tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _tree_to_jax(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


def _serialize_key(key):
    return np.asarray(jax.random.key_data(key))


def _deserialize_key(key_data):
    return jax.random.wrap_key_data(jnp.asarray(key_data))


def _checkpoint_path(checkpoint_dir: Path, run_name: str, suffix: str) -> Path:
    return checkpoint_dir / f"{run_name}-{suffix}.pkl"


def save_checkpoint(
    checkpoint_dir: Path,
    run_name: str,
    suffix: str,
    cfg: SNNConfig,
    epoch: int,
    params,
    noiser_params,
    data_key,
    prefetch_batch_keys,
    global_update,
    ema_success,
    best_test_acc,
    best_epoch,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cfg": asdict(cfg),
        "epoch": epoch,
        "params": _tree_to_numpy(params),
        "noiser_params": _tree_to_numpy(noiser_params),
        "data_key": _serialize_key(data_key),
        "prefetch_batch_keys": [_serialize_key(k) for k in prefetch_batch_keys],
        "global_update": global_update,
        "ema_success": ema_success,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "saved_at": time.time(),
    }

    path = _checkpoint_path(checkpoint_dir, run_name, suffix)
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_checkpoint(path: str):
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    payload["params"] = _tree_to_jax(payload["params"])
    payload["noiser_params"] = _tree_to_jax(payload["noiser_params"])
    payload["data_key"] = _deserialize_key(payload["data_key"])
    payload["prefetch_batch_keys"] = [
        _deserialize_key(k) for k in payload.get("prefetch_batch_keys", [])
    ]
    return payload


def write_metric(metrics_path: Path, record: dict):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def compute_fitness(spike_counts, labels):
    """Negative cross-entropy fitness (higher = better).

    Args:
        spike_counts: [B, n_classes] output spike counts used as logits
        labels: [B] integer class labels

    Returns:
        scalar fitness value
    """
    log_probs = jax.nn.log_softmax(spike_counts)
    ce = -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])
    return -ce


def _sigma_action(cfg: SNNConfig, ema_success: float):
    lower = cfg.sigma_target_success - cfg.sigma_success_tolerance
    upper = cfg.sigma_target_success + cfg.sigma_success_tolerance
    if ema_success > upper:
        return "grow", cfg.sigma_growth
    if ema_success < lower:
        return "decay", cfg.sigma_decay
    return "hold", 1.0


def summarize_output_activity(outputs):
    return {
        "output_nonzero_fraction": float(jnp.mean(outputs != 0)),
        "output_class_variance_mean": float(jnp.mean(jnp.var(outputs, axis=-1))),
    }


def _block_tree(value):
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


class ProfilingController:
    STARTUP_SNAPSHOT_LABELS = {
        "train-begin",
        "model-init-complete",
        "dataset-loaded",
        "prefetch-ready",
        "start-metric-written",
        "jit-forward-eval-ready",
        "population-scores-ready",
        "jit-update-ready",
    }
    STEADY_STATE_SNAPSHOT_LABELS = {
        "steady-state-forward-eval-ready",
        "steady-state-population-scores-ready",
        "steady-state-jit-update-ready",
        "steady-state-eval-ready",
    }

    def __init__(self, cfg: SNNConfig, log_dir: Path):
        self.cfg = cfg
        self.enabled = cfg.profile_mode != "off"
        self.sync_timings = cfg.profile_sync_timings
        self.log_dir = log_dir.resolve()
        self.run_name = cfg.run_name
        self.start_time = time.time()
        self.startup_path = self.log_dir / f"{cfg.run_name}.startup.jsonl"
        self.summary_path = self.log_dir / f"{cfg.run_name}.profile-summary.json"
        self.trace_root = (
            Path(cfg.profile_trace_dir).resolve()
            if cfg.profile_trace_dir
            else (self.log_dir / "traces").resolve()
        )
        self.profile_root = (self.log_dir / "profiles").resolve()
        self.profile_run_root = self.profile_root / cfg.run_name
        self.trace_run_root = self.trace_root / cfg.run_name
        self.snapshot_count = 0
        self.startup_events = []
        self.overall_stage_timings = defaultdict(list)
        self.eval_timings = []
        self.eval_once_recorded = False
        self.server_port = cfg.profile_server_port
        self.server_started = False
        self.startup_trace_started = False
        self.startup_trace_stopped = False
        self.startup_trace_dir = self.trace_run_root / "startup"
        self.steady_trace_started = False
        self.steady_trace_stopped = False
        self.steady_trace_dir = self.trace_run_root / "steady_state"
        self.steady_trace_start_update = cfg.profile_warmup_updates
        self.steady_trace_stop_update = cfg.profile_warmup_updates + cfg.profile_updates_window
        self._startup_trace_active = False
        self._steady_trace_active = False
        if self.enabled:
            self.profile_run_root.mkdir(parents=True, exist_ok=True)
            self.trace_run_root.mkdir(parents=True, exist_ok=True)

    def _write_startup_record(self, record: dict):
        self.startup_path.parent.mkdir(parents=True, exist_ok=True)
        with self.startup_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def log_startup(self, label: str, epoch: int | None = None, global_update: int | None = None):
        if not self.enabled:
            return
        timestamp = time.time()
        record = {
            "label": label,
            "timestamp": timestamp,
            "elapsed_s": timestamp - self.start_time,
            "epoch": epoch,
            "global_update": global_update,
        }
        self.startup_events.append(record)
        self._write_startup_record(record)
        print(
            f"[startup {time.strftime('%Y-%m-%d %H:%M:%S')}] {label}",
            flush=True,
        )

    def maybe_start_server(self, epoch: int | None = None, global_update: int | None = None):
        if not self.enabled or self.server_port is None or self.server_started:
            return
        try:
            jax.profiler.start_server(self.server_port)
            self.server_started = True
            self.log_startup(
                f"profiler server listening on port {self.server_port}",
                epoch=epoch,
                global_update=global_update,
            )
        except Exception as exc:
            self.log_startup(
                f"profiler server failed: {exc}",
                epoch=epoch,
                global_update=global_update,
            )

    def _capture_snapshot(self, label: str, epoch: int | None = None, global_update: int | None = None):
        if not self.enabled or self.snapshot_count >= self.cfg.profile_max_snapshots:
            return
        self.snapshot_count += 1
        path = self.profile_run_root / f"{self.snapshot_count:03d}-{label}.pb"
        self.log_startup(
            f"{label}: saving memory profile -> {path}",
            epoch=epoch,
            global_update=global_update,
        )
        try:
            jax.profiler.save_device_memory_profile(str(path))
            self.log_startup(
                f"{label}: memory profile saved",
                epoch=epoch,
                global_update=global_update,
            )
        except Exception as exc:
            self.log_startup(
                f"{label}: memory profile failed: {exc}",
                epoch=epoch,
                global_update=global_update,
            )

    def maybe_snapshot(self, label: str, epoch: int | None = None, global_update: int | None = None):
        if not self.enabled:
            return
        if label in self.STARTUP_SNAPSHOT_LABELS:
            self._capture_snapshot(label, epoch=epoch, global_update=global_update)
        elif self.label_for_steady_state(label) in self.STEADY_STATE_SNAPSHOT_LABELS:
            self._capture_snapshot(
                self.label_for_steady_state(label),
                epoch=epoch,
                global_update=global_update,
            )

    def maybe_block(self, label: str, value, epoch: int | None = None, global_update: int | None = None):
        if self.enabled and self.sync_timings:
            self.log_startup(
                f"{label}: waiting for device",
                epoch=epoch,
                global_update=global_update,
            )
            _block_tree(value)
            self.log_startup(
                f"{label}: device ready",
                epoch=epoch,
                global_update=global_update,
            )
        return value

    def maybe_start_startup_trace(self, epoch: int | None = None, global_update: int | None = None):
        if not self.enabled or self.cfg.profile_mode not in {"startup", "full"} or self.startup_trace_started:
            return
        self.startup_trace_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(self.startup_trace_dir))
        self.startup_trace_started = True
        self._startup_trace_active = True
        self.log_startup(
            f"startup trace started -> {self.startup_trace_dir}",
            epoch=epoch,
            global_update=global_update,
        )

    def finish_startup_trace(self, label: str, epoch: int | None = None, global_update: int | None = None):
        if self._startup_trace_active and not self.startup_trace_stopped:
            jax.profiler.stop_trace()
            self._startup_trace_active = False
            self.startup_trace_stopped = True
            self.log_startup(
                f"startup trace stopped at {label}",
                epoch=epoch,
                global_update=global_update,
            )

    def maybe_start_steady_state_trace(self, global_update: int, epoch: int | None = None):
        if (
            not self.enabled
            or self.cfg.profile_mode not in {"steady_state", "full"}
            or self.steady_trace_started
            or global_update != self.steady_trace_start_update
        ):
            return
        self.steady_trace_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(self.steady_trace_dir))
        self.steady_trace_started = True
        self._steady_trace_active = True
        self.log_startup(
            f"steady-state trace started -> {self.steady_trace_dir}",
            epoch=epoch,
            global_update=global_update,
        )

    def finish_steady_state_trace(self, global_update: int, epoch: int | None = None):
        if self._steady_trace_active and not self.steady_trace_stopped and global_update >= self.steady_trace_stop_update:
            jax.profiler.stop_trace()
            self._steady_trace_active = False
            self.steady_trace_stopped = True
            self.log_startup(
                "steady-state trace stopped",
                epoch=epoch,
                global_update=global_update,
            )

    def finish_all_traces(self, label: str, epoch: int | None = None, global_update: int | None = None):
        self.finish_startup_trace(label, epoch=epoch, global_update=global_update)
        if self._steady_trace_active and not self.steady_trace_stopped:
            jax.profiler.stop_trace()
            self._steady_trace_active = False
            self.steady_trace_stopped = True
            self.log_startup(
                f"steady-state trace stopped at {label}",
                epoch=epoch,
                global_update=global_update,
            )

    def label_for_steady_state(self, label: str) -> str:
        return f"steady-state-{label}"

    def stage_timed_call(
        self,
        stage: str,
        fn,
        *,
        ready_value=None,
        epoch: int | None = None,
        global_update: int | None = None,
    ):
        t0 = time.perf_counter()
        result = fn()
        if self.enabled and self.sync_timings:
            value = ready_value(result) if callable(ready_value) else (ready_value if ready_value is not None else result)
            self.maybe_block(stage, value, epoch=epoch, global_update=global_update)
        dt = time.perf_counter() - t0
        self.record_stage_timing(stage, dt)
        return result, dt

    def record_stage_timing(self, stage: str, dt: float):
        if self.enabled:
            self.overall_stage_timings[stage].append(dt)

    def record_eval_timing(self, total_s: float, n_chunks: int):
        if self.enabled:
            self.eval_timings.append(
                {
                    "total_s": total_s,
                    "chunk_mean_s": (total_s / n_chunks) if n_chunks > 0 else 0.0,
                }
            )

    def aggregate_epoch_timings(self, epoch_timings: dict[str, list[float]], eval_total_s: float | None = None):
        if not self.enabled or not epoch_timings.get("total_update_s"):
            return {}

        def stats(values):
            arr = np.asarray(values, dtype=np.float64)
            return float(arr.mean()), float(np.median(arr)), float(arr.max())

        total_mean, total_median, total_max = stats(epoch_timings["total_update_s"])
        result = {
            "timing_total_update_mean_s": total_mean,
            "timing_total_update_median_s": total_median,
            "timing_total_update_max_s": total_max,
        }
        for stage in [
            "sample_encode_s",
            "prefix_cache_s",
            "forward_eval_s",
            "population_score_s",
            "update_s",
            "post_update_stats_s",
        ]:
            if epoch_timings.get(stage):
                mean, median, max_v = stats(epoch_timings[stage])
                prefix = stage.removesuffix("_s")
                result[f"timing_{prefix}_mean_s"] = mean
                if stage in {"population_score_s"}:
                    result[f"timing_{prefix}_median_s"] = median
                    result[f"timing_{prefix}_max_s"] = max_v
                result[f"timing_{prefix}_frac"] = mean / total_mean if total_mean > 0 else 0.0
        if eval_total_s is not None:
            result["timing_eval_frac_epoch"] = eval_total_s / (eval_total_s + sum(epoch_timings["total_update_s"])) if (eval_total_s + sum(epoch_timings["total_update_s"])) > 0 else 0.0
        return result

    def write_summary(self):
        if not self.enabled:
            return

        def aggregate(values):
            if not values:
                return None
            arr = np.asarray(values, dtype=np.float64)
            return {
                "mean_s": float(arr.mean()),
                "median_s": float(np.median(arr)),
                "max_s": float(arr.max()),
            }

        stage_summary = {
            stage: aggregate(values)
            for stage, values in self.overall_stage_timings.items()
            if values
        }
        total_mean = stage_summary.get("total_update_s", {}).get("mean_s", 0.0) if stage_summary.get("total_update_s") else 0.0
        bottlenecks = []
        for stage in [
            "population_score_s",
            "update_s",
            "sample_encode_s",
            "prefix_cache_s",
            "forward_eval_s",
            "post_update_stats_s",
        ]:
            summary = stage_summary.get(stage)
            if summary and total_mean > 0:
                bottlenecks.append(
                    {
                        "stage": stage.removesuffix("_s"),
                        "fraction": summary["mean_s"] / total_mean,
                    }
                )
        bottlenecks.sort(key=lambda item: item["fraction"], reverse=True)

        payload = {
            "profile_mode": self.cfg.profile_mode,
            "startup_events": self.startup_events,
            "startup_trace_captured": self.startup_trace_started,
            "startup_trace_stopped": self.startup_trace_stopped,
            "startup_trace_dir": str(self.startup_trace_dir) if self.startup_trace_started else None,
            "steady_state_trace_captured": self.steady_trace_started,
            "steady_state_trace_stopped": self.steady_trace_stopped,
            "steady_state_trace_dir": str(self.steady_trace_dir) if self.steady_trace_started else None,
            "profiler_server_port": self.server_port if self.server_started else None,
            "memory_snapshots_written": self.snapshot_count,
            "memory_profile_dir": str(self.profile_run_root),
            "startup_jsonl": str(self.startup_path),
            "stage_timing_summary": stage_summary,
            "eval_timing_summary": {
                "count": len(self.eval_timings),
                "mean_total_s": float(np.mean([item["total_s"] for item in self.eval_timings])) if self.eval_timings else None,
                "mean_chunk_s": float(np.mean([item["chunk_mean_s"] for item in self.eval_timings])) if self.eval_timings else None,
                "fraction_vs_mean_update": (
                    float(np.mean([item["total_s"] for item in self.eval_timings])) / total_mean
                    if self.eval_timings and total_mean > 0
                    else None
                ),
            },
            "bottleneck_ranking": bottlenecks,
        }
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with self.summary_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)


def train(cfg: SNNConfig = None):
    """Run EGGROLL training.

    Args:
        cfg: hyperparameter config, uses defaults if None

    Returns:
        (frozen_params, params, noiser_params, test_accuracy)
    """
    if cfg is None:
        cfg = SNNConfig()

    log_dir = Path(cfg.log_dir).resolve()
    checkpoint_dir = Path(cfg.checkpoint_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / f"{cfg.run_name}.metrics.jsonl"
    summary_path = log_dir / f"{cfg.run_name}.summary.json"
    profiler = ProfilingController(cfg, log_dir)
    profiler.log_startup(f"train() begin run_name={cfg.run_name}")
    profiler.maybe_snapshot("train-begin")
    profiler.maybe_start_server()
    profiler.maybe_start_startup_trace()

    N = cfg.pop_size
    assert N % 2 == 0, "Population size must be even (antithetical sampling)"

    # PRNG keys
    key = jax.random.key(cfg.seed)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    data_key = jax.random.fold_in(key, 2)

    model_cls = get_model_cls(cfg.model_name)

    # Initialize model
    profiler.log_startup("initializing model")
    frozen_params, params, scan_map, es_map = model_cls.rand_init(model_key, cfg)
    profiler.log_startup("model init complete")
    profiler.maybe_snapshot("model-init-complete")
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)

    # Initialize EGGROLL noiser
    profiler.log_startup("initializing noiser")
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        cfg.sigma,
        cfg.lr,
        solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999},
        rank=cfg.rank,
        use_batched_update=cfg.use_batched_update,
        fitness_shaping=cfg.fitness_shaping,
    )
    profiler.log_startup("noiser init complete")

    start_epoch = 0
    global_update = 0
    ema_success = None
    best_test_acc = float("-inf")
    best_epoch = None
    pending_batch_keys = []

    if cfg.resume_from:
        profiler.log_startup(f"loading checkpoint {cfg.resume_from}")
        checkpoint = load_checkpoint(cfg.resume_from)
        params = checkpoint["params"]
        noiser_params = checkpoint["noiser_params"]
        data_key = checkpoint["data_key"]
        pending_batch_keys = checkpoint.get("prefetch_batch_keys", [])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_update = int(checkpoint.get("global_update", start_epoch))
        ema_success = checkpoint.get("ema_success")
        best_test_acc = float(checkpoint.get("best_test_acc", float("-inf")))
        best_epoch = checkpoint.get("best_epoch")
        print(f"Resuming from {cfg.resume_from} at epoch {start_epoch}")
        write_metric(
            metrics_path,
            {
                "event": "resume",
                "resume_from": str(cfg.resume_from),
                "start_epoch": start_epoch,
                "global_update": global_update,
                "timestamp": time.time(),
            },
        )

    # Load dataset
    profiler.log_startup("loading dataset")
    dataset_spec = get_dataset_spec(cfg)
    train_data, train_labels, test_data, test_labels = dataset_spec.loader()
    profiler.log_startup(
        f"dataset loaded train={train_data.shape} test={test_data.shape}"
    )
    profiler.maybe_snapshot("dataset-loaded")
    encode_batch_fn = dataset_spec.encoder
    default_n_inputs = dataset_spec.n_inputs
    default_channels = dataset_spec.in_channels
    default_img = dataset_spec.image_size
    if cfg.in_channels != default_channels or cfg.image_size != default_img:
        print(
            f"Warning: cfg image settings ({cfg.in_channels}ch, {cfg.image_size}px) "
            f"differ from dataset defaults ({default_channels}ch, {default_img}px)."
        )
    if cfg.n_inputs != default_n_inputs:
        print(
            f"Warning: cfg.n_inputs={cfg.n_inputs} differs from dataset default "
            f"{default_n_inputs}. Using cfg value."
        )
    _timesteps = cfg.timesteps
    _batch_size = cfg.batch_size
    _chunk_size = cfg.chunk_size
    _prefetch_depth = 2

    @jax.jit
    def sample_and_encode(images, labels, key):
        k1, k2, k3 = jax.random.split(key, 3)
        indices = jax.random.choice(k1, images.shape[0], shape=(_batch_size,), replace=False)
        batch_imgs = images[indices]
        if _do_augment:  # Python bool — evaluated at trace time, dead-code-eliminated if False
            batch_imgs = _augment_cifar(batch_imgs, k3)
        spikes = encode_batch_fn(batch_imgs, _timesteps, k2)
        return spikes, labels[indices]

    print(f"Dataset: {cfg.dataset} | train: {train_data.shape}, test: {test_data.shape}")
    print(f"Model: {cfg.model_name}")
    if cfg.model_name == "mlp_snn":
        print(f"Architecture: {cfg.n_inputs}-{cfg.hidden_size}-{cfg.hidden_size}-{cfg.n_classes}")
    else:
        print(
            "Architecture: spiking_resnet18 "
            f"stages={list(cfg.resnet_block_counts)} "
            f"channels={[cfg.resnet_channels_base * (2 ** i) for i in range(4)]} "
            f"in_channels={cfg.in_channels} norm={cfg.resnet_norm}:{cfg.resnet_norm_groups} "
            f"conv_es={cfg.conv_es_mode} "
            f"classes={cfg.n_classes}"
        )
    print(
        f"EGGROLL: pop={N}, rank={cfg.rank}, sigma={cfg.sigma}, lr={cfg.lr}, "
        f"shape={cfg.fitness_shaping}, batched_update={cfg.use_batched_update}"
    )
    print(
        "Sigma control: "
        f"target={cfg.sigma_target_success:.3f}±{cfg.sigma_success_tolerance:.3f} "
        f"grow={cfg.sigma_growth:.3f} decay={cfg.sigma_decay:.3f} "
        f"ema={cfg.sigma_ema_decay:.3f} clip=[{cfg.sigma_min:.4f}, {cfg.sigma_max:.4f}]"
    )
    print(
        "Selective perturbation: "
        f"enabled={cfg.selective_stage_perturbation} "
        f"schedule={cfg.stage_perturbation_schedule} "
        f"early_fraction={cfg.stage_perturbation_early_fraction:.2f} "
        f"full_interval={cfg.stage_perturbation_full_epoch_interval}"
    )
    total_updates = cfg.num_epochs * cfg.updates_per_epoch
    total_samples = total_updates * cfg.batch_size
    if cfg.dataset == "cifar10":
        print(
            f"Budget: {cfg.num_epochs} epochs × {cfg.updates_per_epoch} upd/epoch "
            f"= {total_updates} total updates | "
            f"~{total_samples:,} samples ({total_samples / 50000:.1f}× CIFAR train set)"
        )
    else:
        print(
            f"Budget: {cfg.num_epochs} epochs × {cfg.updates_per_epoch} upd/epoch "
            f"= {total_updates} total updates | ~{total_samples:,} samples"
        )
    print(f"Run: {cfg.run_name} | metrics: {metrics_path} | checkpoints: {checkpoint_dir}")

    # Augmentation: only for CIFAR-10, evaluated at Python trace time (JIT-safe constant)
    _do_augment = cfg.augment and cfg.dataset == "cifar10"

    # Precompute layer 1 base: x @ W1.T for all timesteps (shared across population)
    profiler.log_startup("building jit wrappers")
    @jax.jit
    def compute_l1_base(params, x):
        """Compute x @ W1.T once, returns [T, B, hidden]."""
        x_t = jnp.transpose(x, (1, 0, 2))  # [T, B, n_inputs]
        W1 = params["linear1"]["weight"]     # [hidden, n_inputs]
        return x_t @ W1.T                    # [T, B, hidden]

    use_resnet_running_stats_norm = (
        cfg.model_name == "spiking_resnet18" and cfg.resnet_norm in {"batch", "bntt"}
    )

    if use_resnet_running_stats_norm:
        jit_forward = jax.jit(
            jax.vmap(
                lambda n, p, i, x, l1b: model_cls.forward(
                    EggRoll,
                    frozen_noiser_params,
                    n,
                    frozen_params,
                    p,
                    es_tree_key,
                    i,
                    x,
                    l1b,
                    norm_training=True,
                ),
                in_axes=(None, None, 0, None, None),
            )
        )
        jit_forward_eval = jax.jit(
            lambda n, p, x: model_cls.forward(
                EggRoll,
                frozen_noiser_params,
                n,
                frozen_params,
                p,
                es_tree_key,
                None,
                x,
                norm_training=False,
            )
        )
        jit_forward_train_with_bn_stats = jax.jit(
            lambda n, p, x: model_cls.forward_train_with_bn_stats(
                EggRoll,
                frozen_noiser_params,
                n,
                frozen_params,
                p,
                es_tree_key,
                None,
                x,
            )
        )
    else:
        jit_forward = jax.jit(
            jax.vmap(
                lambda n, p, i, x, l1b: model_cls.forward(
                    EggRoll,
                    frozen_noiser_params,
                    n,
                    frozen_params,
                    p,
                    es_tree_key,
                    i,
                    x,
                    l1b,
                ),
                in_axes=(None, None, 0, None, None),
            )
        )

        jit_forward_eval = jax.jit(
            lambda n, p, x: model_cls.forward(
                EggRoll,
                frozen_noiser_params,
                n,
                frozen_params,
                p,
                es_tree_key,
                None,
                x,
            )
        )
        jit_forward_train_with_bn_stats = None

    selective_supported = (
        cfg.selective_stage_perturbation and cfg.model_name == "spiking_resnet18"
    )
    selective_phase_info = {}
    if selective_supported:
        perturb_group_map = frozen_params["perturb_group_map"]
        for phase in model_cls.PHASES:
            plan = {
                "early_selective": model_cls.resolve_selective_plan(
                    0, cfg.num_epochs, cfg
                ),
                "mid_selective": {
                    "phase": "mid_selective",
                    "active_groups": ("stage2", "stage3", "head"),
                    "cache_split": "after_stage1",
                },
                "full_model_refresh": {
                    "phase": "full_model_refresh",
                    "active_groups": (
                        "stem",
                        "stage0",
                        "stage1",
                        "stage2",
                        "stage3",
                        "head",
                    ),
                    "cache_split": None,
                },
            }[phase]
            active_es_map = model_cls.build_active_es_map(
                es_map, perturb_group_map, plan["active_groups"]
            )
            selective_phase_info[phase] = {
                **plan,
                "active_es_map": active_es_map,
                "active_param_fraction": model_cls.active_param_fraction(
                    es_map, active_es_map
                ),
            }

        jit_prefix_after_stage1 = jax.jit(
            lambda n, p, x: model_cls.forward_prefix_after_stage1(
                EggRoll,
                frozen_noiser_params,
                n,
                frozen_params,
                p,
                es_tree_key,
                x,
                norm_training=True,
            )
        )
        jit_prefix_after_stage2 = jax.jit(
            lambda n, p, x: model_cls.forward_prefix_after_stage2(
                EggRoll,
                frozen_noiser_params,
                n,
                frozen_params,
                p,
                es_tree_key,
                x,
                norm_training=True,
            )
        )
        jit_forward_selective_after_stage1 = jax.jit(
            jax.vmap(
                lambda n, p, i, prefix_x: model_cls.forward_suffix_after_stage1(
                    EggRoll,
                    frozen_noiser_params,
                    n,
                    frozen_params,
                    p,
                    es_tree_key,
                    i,
                    prefix_x,
                    norm_training=True,
                ),
                in_axes=(None, None, 0, None),
            )
        )
        jit_forward_selective_after_stage2 = jax.jit(
            jax.vmap(
                lambda n, p, i, prefix_x: model_cls.forward_suffix_after_stage2(
                    EggRoll,
                    frozen_noiser_params,
                    n,
                    frozen_params,
                    p,
                    es_tree_key,
                    i,
                    prefix_x,
                    norm_training=True,
                ),
                in_axes=(None, None, 0, None),
            )
        )

        jit_update_selective = {
            phase: jax.jit(
                lambda n, p, f, i, active_es_map=phase_info["active_es_map"]: EggRoll.do_updates(
                    frozen_noiser_params,
                    n,
                    p,
                    es_tree_key,
                    f,
                    i,
                    active_es_map,
                ),
                donate_argnums=(0, 1),
            )
            for phase, phase_info in selective_phase_info.items()
        }
    else:
        jit_prefix_after_stage1 = None
        jit_prefix_after_stage2 = None
        jit_forward_selective_after_stage1 = None
        jit_forward_selective_after_stage2 = None
        jit_update_selective = {}

    # JIT-compiled parameter update
    jit_update = jax.jit(
        lambda n, p, f, i: EggRoll.do_updates(
            frozen_noiser_params, n, p, es_tree_key, f, i, es_map
        ),
        donate_argnums=(0, 1),
    )

    # Evaluate test set in fixed-size chunks (same batch size as training).
    # Note: remainder samples are intentionally dropped to avoid shape-triggered recompiles.
    eval_test_size = test_data.shape[0]
    if cfg.num_test_eval_samples and cfg.num_test_eval_samples > 0:
        eval_test_size = min(eval_test_size, cfg.num_test_eval_samples)
    n_test_chunks = eval_test_size // cfg.batch_size

    if _chunk_size > 0 and _chunk_size < N:
        _score_pad = (-N) % _chunk_size
        _padded_size = N + _score_pad
        _chunk_starts = jnp.arange(0, _padded_size, _chunk_size, dtype=jnp.int32)
    else:
        _score_pad = 0
        _padded_size = N
        _chunk_starts = None

    @jax.jit
    def score_population_full(noiser_params, params, iterinfo, x, l1b, y):
        pop_out = jit_forward(noiser_params, params, iterinfo, x, l1b)
        return jax.vmap(compute_fitness, in_axes=(0, None))(pop_out, y)

    if selective_supported:

        @jax.jit
        def score_population_selective_after_stage1_full(
            noiser_params, params, iterinfo, prefix_x, y
        ):
            pop_out = jit_forward_selective_after_stage1(
                noiser_params, params, iterinfo, prefix_x
            )
            return jax.vmap(compute_fitness, in_axes=(0, None))(pop_out, y)

        @jax.jit
        def score_population_selective_after_stage2_full(
            noiser_params, params, iterinfo, prefix_x, y
        ):
            pop_out = jit_forward_selective_after_stage2(
                noiser_params, params, iterinfo, prefix_x
            )
            return jax.vmap(compute_fitness, in_axes=(0, None))(pop_out, y)

    if _chunk_starts is not None:

        @jax.jit
        def score_population_chunked(
            noiser_params, params, epoch_ids, thread_ids, x, l1b, y
        ):
            if _score_pad:
                epoch_ids = jnp.pad(epoch_ids, (0, _score_pad), mode="edge")
                thread_ids = jnp.pad(thread_ids, (0, _score_pad), mode="edge")

            def score_chunk(start):
                idx = jnp.arange(_chunk_size, dtype=jnp.int32) + start
                c_iter = (epoch_ids[idx], thread_ids[idx])
                return score_population_full(noiser_params, params, c_iter, x, l1b, y)

            return jax.lax.map(score_chunk, _chunk_starts).reshape(-1)[:N]

        if selective_supported:

            @jax.jit
            def score_population_selective_after_stage1_chunked(
                noiser_params, params, epoch_ids, thread_ids, prefix_x, y
            ):
                if _score_pad:
                    epoch_ids = jnp.pad(epoch_ids, (0, _score_pad), mode="edge")
                    thread_ids = jnp.pad(thread_ids, (0, _score_pad), mode="edge")

                def score_chunk(start):
                    idx = jnp.arange(_chunk_size, dtype=jnp.int32) + start
                    c_iter = (epoch_ids[idx], thread_ids[idx])
                    return score_population_selective_after_stage1_full(
                        noiser_params, params, c_iter, prefix_x, y
                    )

                return jax.lax.map(score_chunk, _chunk_starts).reshape(-1)[:N]

            @jax.jit
            def score_population_selective_after_stage2_chunked(
                noiser_params, params, epoch_ids, thread_ids, prefix_x, y
            ):
                if _score_pad:
                    epoch_ids = jnp.pad(epoch_ids, (0, _score_pad), mode="edge")
                    thread_ids = jnp.pad(thread_ids, (0, _score_pad), mode="edge")

                def score_chunk(start):
                    idx = jnp.arange(_chunk_size, dtype=jnp.int32) + start
                    c_iter = (epoch_ids[idx], thread_ids[idx])
                    return score_population_selective_after_stage2_full(
                        noiser_params, params, c_iter, prefix_x, y
                    )

                return jax.lax.map(score_chunk, _chunk_starts).reshape(-1)[:N]
    profiler.log_startup("jit wrappers ready")

    def eval_test():
        if n_test_chunks == 0:
            return float("nan"), None

        eval_inputs = test_data[: n_test_chunks * cfg.batch_size]
        eval_labels = test_labels[: n_test_chunks * cfg.batch_size]

        @jax.jit
        def eval_batches(noiser_params, params, images, labels):
            def eval_one(i):
                start = i * cfg.batch_size
                te_chunk = jax.lax.dynamic_slice_in_dim(images, start, cfg.batch_size, axis=0)
                te_key = jax.random.fold_in(jax.random.key(999), i)
                te_spikes = encode_batch_fn(te_chunk, cfg.timesteps, te_key)
                out = jit_forward_eval(noiser_params, params, te_spikes)
                return jnp.argmax(out, axis=-1)

            preds = jax.lax.map(eval_one, jnp.arange(n_test_chunks, dtype=jnp.int32)).reshape(-1)
            return jnp.mean(preds == labels)

        def run_eval():
            return eval_batches(noiser_params, params, eval_inputs, eval_labels)

        if profiler.enabled:
            eval_acc, total_s = profiler.stage_timed_call(
                "eval_test_s",
                run_eval,
                epoch=epoch_for_eval,
                global_update=global_update_for_eval,
            )
            profiler.record_eval_timing(total_s, n_test_chunks)
            if profile_snapshot_label:
                profiler.maybe_snapshot(
                    profile_snapshot_label,
                    epoch=epoch_for_eval,
                    global_update=global_update_for_eval,
                )
            return float(eval_acc), {
                "eval_test_total_s": total_s,
                "eval_test_chunk_mean_s": total_s / n_test_chunks if n_test_chunks > 0 else 0.0,
            }

        return float(run_eval()), None

    epoch_for_eval = None
    global_update_for_eval = None
    profile_snapshot_label = None

    def build_prefetch_queue(start_key, queued_keys=None):
        profiler.log_startup(
            f"prefetch build begin depth={_prefetch_depth} queued={len(queued_keys or [])}"
        )
        queue = deque()
        next_key = start_key
        batch_keys = list(queued_keys or [])

        while len(batch_keys) < _prefetch_depth:
            next_key, batch_key = jax.random.split(next_key)
            batch_keys.append(batch_key)

        for batch_key in batch_keys:
            profiler.log_startup("prefetch sample_and_encode begin")
            x_batch, y_batch = sample_and_encode(train_data, train_labels, batch_key)
            profiler.maybe_block("prefetch sample_and_encode", x_batch)
            profiler.maybe_block("prefetch labels", y_batch)
            queue.append((batch_key, x_batch, y_batch))
            profiler.log_startup("prefetch batch enqueued")

        profiler.log_startup("prefetch build complete")
        return queue, next_key

    def pop_prefetched_batch(queue, next_key):
        _, x_batch, y_batch = queue.popleft()
        next_key, batch_key = jax.random.split(next_key)
        profiler.log_startup("next sample_and_encode begin", epoch=epoch_context, global_update=global_update)
        next_x, next_y = sample_and_encode(train_data, train_labels, batch_key)
        profiler.maybe_block("next sample_and_encode", next_x, epoch=epoch_context, global_update=global_update)
        profiler.maybe_block("next labels", next_y, epoch=epoch_context, global_update=global_update)
        profiler.maybe_snapshot("next-batch-ready", epoch=epoch_context, global_update=global_update)
        queue.append((batch_key, next_x, next_y))
        profiler.log_startup("next batch enqueued", epoch=epoch_context, global_update=global_update)
        return x_batch, y_batch, queue, next_key

    epoch_context = None
    profiler.log_startup("starting prefetch queue warmup")
    prefetch_queue, data_key = build_prefetch_queue(data_key, pending_batch_keys)
    profiler.log_startup("prefetch queue ready")
    profiler.maybe_snapshot("prefetch-ready")

    profiler.log_startup("writing start metric")
    write_metric(
        metrics_path,
        {
            "event": "start",
            "run_name": cfg.run_name,
            "cfg": asdict(cfg),
            "sigma_target_success": cfg.sigma_target_success,
            "sigma_success_tolerance": cfg.sigma_success_tolerance,
            "sigma_growth": cfg.sigma_growth,
            "sigma_decay": cfg.sigma_decay,
            "sigma_ema_decay": cfg.sigma_ema_decay,
            "timestamp": time.time(),
            "start_epoch": start_epoch,
            "global_update": global_update,
        },
    )
    profiler.log_startup("start metric written")
    profiler.maybe_snapshot("start-metric-written")

    # Training loop
    t_start = time.time()
    last_completed_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            epoch_context = epoch
            profiler.log_startup(f"epoch {epoch} begin", epoch=epoch, global_update=global_update)
            epoch_timings = defaultdict(list)
            eval_stats = None
            selective_plan = None
            perturbation_phase = "disabled"
            active_stage_groups = ()
            cache_split = None
            active_param_fraction = 1.0
            if selective_supported:
                resolved_plan = model_cls.resolve_selective_plan(epoch, cfg.num_epochs, cfg)
                perturbation_phase = resolved_plan["phase"]
                selective_plan = selective_phase_info[perturbation_phase]
                active_stage_groups = selective_plan["active_groups"]
                cache_split = selective_plan["cache_split"]
                active_param_fraction = selective_plan["active_param_fraction"]
            for _ in range(cfg.updates_per_epoch):
                profiler.maybe_start_steady_state_trace(global_update, epoch=epoch)
                update_start = time.perf_counter()
                (x_batch, y_batch, prefetch_queue, data_key), sample_encode_s = profiler.stage_timed_call(
                    "sample_encode_s",
                    lambda: pop_prefetched_batch(prefetch_queue, data_key),
                    ready_value=lambda result: (result[0], result[1]),
                    epoch=epoch,
                    global_update=global_update,
                )
                epoch_timings["sample_encode_s"].append(sample_encode_s)

                # Build iterinfo for population
                iterinfo = (
                    jnp.full(N, global_update, dtype=jnp.int32),
                    jnp.arange(N),
                )

                # Evaluate base params (no noise) on this batch
                profiler.log_startup("jit_forward_eval begin", epoch=epoch, global_update=global_update)
                if use_resnet_running_stats_norm:
                    (val_out, bn_stats), forward_eval_s = profiler.stage_timed_call(
                        "forward_eval_s",
                        lambda: jit_forward_train_with_bn_stats(
                            noiser_params, params, x_batch
                        ),
                        epoch=epoch,
                        global_update=global_update,
                    )
                    params = model_cls.apply_bn_running_stats(
                        params,
                        bn_stats,
                        cfg.resnet_bntt_momentum
                        if cfg.resnet_norm == "bntt"
                        else cfg.resnet_bn_momentum,
                    )
                else:
                    val_out, forward_eval_s = profiler.stage_timed_call(
                        "forward_eval_s",
                        lambda: jit_forward_eval(noiser_params, params, x_batch),
                        epoch=epoch,
                        global_update=global_update,
                    )
                epoch_timings["forward_eval_s"].append(forward_eval_s)
                if not profiler.startup_trace_stopped:
                    profiler.maybe_snapshot("jit-forward-eval-ready", epoch=epoch, global_update=global_update)
                elif profiler.steady_trace_started and not profiler.steady_trace_stopped:
                    profiler.maybe_snapshot("steady-state-forward-eval-ready", epoch=epoch, global_update=global_update)
                val_fitness = compute_fitness(val_out, y_batch)

                # Precompute layer 1 base matmul when model exposes linear1
                l1_base = None
                if "linear1" in params:
                    l1_base = compute_l1_base(params, x_batch)

                prefix_cache = None
                prefix_cache_s = 0.0
                if selective_supported and cache_split is not None:
                    if cache_split == "after_stage1":
                        prefix_cache, prefix_cache_s = profiler.stage_timed_call(
                            "prefix_cache_s",
                            lambda: jit_prefix_after_stage1(noiser_params, params, x_batch),
                            epoch=epoch,
                            global_update=global_update,
                        )
                    else:
                        prefix_cache, prefix_cache_s = profiler.stage_timed_call(
                            "prefix_cache_s",
                            lambda: jit_prefix_after_stage2(noiser_params, params, x_batch),
                            epoch=epoch,
                            global_update=global_update,
                        )
                    epoch_timings["prefix_cache_s"].append(prefix_cache_s)

                # Evaluate population (with noise)
                if selective_supported and cache_split == "after_stage1":
                    profiler.log_startup(
                        "score_population_selective_after_stage1 begin",
                        epoch=epoch,
                        global_update=global_update,
                    )
                    if _chunk_starts is not None:
                        raw_scores, population_score_s = profiler.stage_timed_call(
                            "population_score_s",
                            lambda: score_population_selective_after_stage1_chunked(
                                noiser_params,
                                params,
                                iterinfo[0],
                                iterinfo[1],
                                prefix_cache,
                                y_batch,
                            ),
                            epoch=epoch,
                            global_update=global_update,
                        )
                    else:
                        raw_scores, population_score_s = profiler.stage_timed_call(
                            "population_score_s",
                            lambda: score_population_selective_after_stage1_full(
                                noiser_params, params, iterinfo, prefix_cache, y_batch
                            ),
                            epoch=epoch,
                            global_update=global_update,
                        )
                elif selective_supported and cache_split == "after_stage2":
                    profiler.log_startup(
                        "score_population_selective_after_stage2 begin",
                        epoch=epoch,
                        global_update=global_update,
                    )
                    if _chunk_starts is not None:
                        raw_scores, population_score_s = profiler.stage_timed_call(
                            "population_score_s",
                            lambda: score_population_selective_after_stage2_chunked(
                                noiser_params,
                                params,
                                iterinfo[0],
                                iterinfo[1],
                                prefix_cache,
                                y_batch,
                            ),
                            epoch=epoch,
                            global_update=global_update,
                        )
                    else:
                        raw_scores, population_score_s = profiler.stage_timed_call(
                            "population_score_s",
                            lambda: score_population_selective_after_stage2_full(
                                noiser_params, params, iterinfo, prefix_cache, y_batch
                            ),
                            epoch=epoch,
                            global_update=global_update,
                        )
                elif _chunk_starts is not None:
                    profiler.log_startup("score_population_chunked begin", epoch=epoch, global_update=global_update)
                    raw_scores, population_score_s = profiler.stage_timed_call(
                        "population_score_s",
                        lambda: score_population_chunked(
                            noiser_params,
                            params,
                            iterinfo[0],
                            iterinfo[1],
                            x_batch,
                            l1_base,
                            y_batch,
                        ),
                        epoch=epoch,
                        global_update=global_update,
                    )
                else:
                    profiler.log_startup("score_population_full begin", epoch=epoch, global_update=global_update)
                    raw_scores, population_score_s = profiler.stage_timed_call(
                        "population_score_s",
                        lambda: score_population_full(
                            noiser_params, params, iterinfo, x_batch, l1_base, y_batch
                        ),
                        epoch=epoch,
                        global_update=global_update,
                    )
                epoch_timings["population_score_s"].append(population_score_s)
                if not profiler.startup_trace_stopped:
                    profiler.maybe_snapshot("population-scores-ready", epoch=epoch, global_update=global_update)
                elif profiler.steady_trace_started and not profiler.steady_trace_stopped:
                    profiler.maybe_snapshot("steady-state-population-scores-ready", epoch=epoch, global_update=global_update)

                fitnesses = EggRoll.convert_fitnesses(
                    frozen_noiser_params, noiser_params, raw_scores
                )
                raw_score_std = float(jnp.std(raw_scores))

                # Update parameters
                profiler.log_startup("jit_update begin", epoch=epoch, global_update=global_update)
                (noiser_params, params), update_s = profiler.stage_timed_call(
                    "update_s",
                    lambda: (
                        jit_update_selective[perturbation_phase](
                            noiser_params, params, fitnesses, iterinfo
                        )
                        if selective_supported
                        else jit_update(noiser_params, params, fitnesses, iterinfo)
                    ),
                    ready_value=lambda result: result[1],
                    epoch=epoch,
                    global_update=global_update,
                )
                epoch_timings["update_s"].append(update_s)
                if not profiler.startup_trace_stopped:
                    profiler.maybe_snapshot("jit-update-ready", epoch=epoch, global_update=global_update)
                elif profiler.steady_trace_started and not profiler.steady_trace_stopped:
                    profiler.maybe_snapshot("steady-state-jit-update-ready", epoch=epoch, global_update=global_update)
                profiler.finish_startup_trace("jit-update-ready", epoch=epoch, global_update=global_update)

                # 1/5th success rule: adapt sigma based on fraction beating baseline
                stats_t0 = time.perf_counter()
                n_better = int(jnp.sum(raw_scores > val_fitness))
                success_rate = n_better / N

                if ema_success is None:
                    ema_success = success_rate
                else:
                    ema_success = (
                        cfg.sigma_ema_decay * ema_success
                        + (1.0 - cfg.sigma_ema_decay) * success_rate
                    )

                sigma_action = "hold"
                if global_update >= cfg.sigma_warmup_epochs * cfg.updates_per_epoch:
                    sigma_action, sigma_factor = _sigma_action(cfg, float(ema_success))
                    if sigma_action != "hold":
                        noiser_params["sigma"] = noiser_params["sigma"] * sigma_factor

                noiser_params["sigma"] = jnp.clip(
                    noiser_params["sigma"], cfg.sigma_min, cfg.sigma_max
                )
                global_update += 1
                profiler.finish_steady_state_trace(global_update, epoch=epoch)
                post_update_stats_s = time.perf_counter() - stats_t0
                epoch_timings["post_update_stats_s"].append(post_update_stats_s)
                profiler.record_stage_timing("post_update_stats_s", post_update_stats_s)
                total_update_s = time.perf_counter() - update_start
                epoch_timings["total_update_s"].append(total_update_s)
                profiler.record_stage_timing("total_update_s", total_update_s)

            do_log = cfg.log_interval > 0 and epoch % cfg.log_interval == 0
            do_test = cfg.test_interval > 0 and epoch % cfg.test_interval == 0
            do_checkpoint = cfg.checkpoint_interval > 0 and epoch % cfg.checkpoint_interval == 0

            val_acc = float(jnp.mean(jnp.argmax(val_out, axis=-1) == y_batch))
            output_activity = summarize_output_activity(val_out)
            elapsed = time.time() - t_start
            eps = (epoch - start_epoch + 1) / elapsed if elapsed > 0 else 0.0
            updates_per_s = global_update / elapsed if elapsed > 0 and global_update > 0 else 0.0
            avg_update_s = elapsed / global_update if global_update > 0 else 0.0
            avg_epoch_s = elapsed / (epoch - start_epoch + 1) if epoch >= start_epoch else 0.0
            epoch_for_eval = epoch
            global_update_for_eval = global_update
            profile_snapshot_label = (
                "steady-state-eval-ready"
                if profiler.steady_trace_started and not profiler.steady_trace_stopped
                else None
            )
            test_acc, eval_stats = eval_test() if do_test else (None, None)

            if cfg.profile_eval_once and profiler.enabled and not profiler.eval_once_recorded and epoch >= start_epoch:
                eval_once_acc, eval_once_stats = eval_test()
                write_metric(
                    metrics_path,
                    {
                        "event": "eval_profile",
                        "epoch": epoch,
                        "global_update": global_update,
                        "eval_test_acc": eval_once_acc,
                        **(eval_once_stats or {}),
                        "timestamp": time.time(),
                    },
                )
                profiler.eval_once_recorded = True

            record = {
                "event": "epoch",
                "epoch": epoch,
                "global_update": global_update,
                "elapsed_s": elapsed,
                "epochs_per_s": eps,
                "updates_per_s": updates_per_s,
                "avg_update_s": avg_update_s,
                "avg_epoch_s": avg_epoch_s,
                "val_fitness": float(val_fitness),
                "val_acc": val_acc,
                "sigma": float(noiser_params["sigma"]),
                "raw_score_std": raw_score_std,
                "n_better": n_better,
                "pop_size": N,
                "success_rate": success_rate,
                "ema_success": float(ema_success),
                "sigma_action": sigma_action,
                "selective_stage_perturbation": selective_supported,
                "perturbation_phase": perturbation_phase,
                "active_stage_groups": list(active_stage_groups),
                "cache_split": cache_split,
                "active_param_fraction": active_param_fraction,
                **output_activity,
                "test_acc": test_acc,
                "timestamp": time.time(),
            }
            record.update(
                profiler.aggregate_epoch_timings(
                    epoch_timings,
                    eval_total_s=(eval_stats or {}).get("eval_test_total_s"),
                )
            )
            if eval_stats:
                record.update(eval_stats)
            write_metric(metrics_path, record)

            if test_acc is not None and test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                best_path = save_checkpoint(
                    checkpoint_dir,
                    cfg.run_name,
                    "best",
                    cfg,
                    epoch,
                    params,
                    noiser_params,
                    data_key,
                    [item[0] for item in prefetch_queue],
                    global_update,
                    ema_success,
                    best_test_acc,
                    best_epoch,
                )
                print(f"Saved best checkpoint: {best_path}")

            if do_checkpoint:
                last_path = save_checkpoint(
                    checkpoint_dir,
                    cfg.run_name,
                    "last",
                    cfg,
                    epoch,
                    params,
                    noiser_params,
                    data_key,
                    [item[0] for item in prefetch_queue],
                    global_update,
                    ema_success,
                    best_test_acc,
                    best_epoch,
                )
                print(f"Saved checkpoint: {last_path}")

            if do_log:
                test_str = f" | test: {test_acc:.4f}" if test_acc is not None else ""
                print(
                    f"Epoch {epoch:4d}/{cfg.num_epochs} | "
                    f"fitness: {float(val_fitness):.4f} | "
                        f"acc: {val_acc:.4f} | "
                        f"σ: {float(noiser_params['sigma']):.5f} | "
                        f"better: {n_better:4d}/{N} std:{raw_score_std:.5f} ema:{float(ema_success):.3f} | "
                        f"{eps:.1f} ep/s {updates_per_s:.2f} upd/s"
                        f"{test_str}"
                )
            last_completed_epoch = epoch
    except KeyboardInterrupt:
        profiler.finish_all_traces("keyboard-interrupt", epoch=last_completed_epoch, global_update=global_update)
        interrupt_path = save_checkpoint(
            checkpoint_dir,
            cfg.run_name,
            "interrupt",
            cfg,
            last_completed_epoch,
            params,
            noiser_params,
            data_key,
            [item[0] for item in prefetch_queue],
            global_update,
            ema_success,
            best_test_acc,
            best_epoch,
        )
        print(f"\nInterrupted. Saved checkpoint: {interrupt_path}")
        raise
    finally:
        profiler.finish_all_traces("train-finally", epoch=last_completed_epoch, global_update=global_update)

    # Final test accuracy
    epoch_for_eval = last_completed_epoch
    global_update_for_eval = global_update
    profile_snapshot_label = None
    test_acc, _ = eval_test()
    elapsed = time.time() - t_start

    save_checkpoint(
        checkpoint_dir,
        cfg.run_name,
        "last",
        cfg,
        last_completed_epoch,
        params,
        noiser_params,
        data_key,
        [item[0] for item in prefetch_queue],
        global_update,
        ema_success,
        best_test_acc,
        best_epoch,
    )

    summary = {
        "run_name": cfg.run_name,
        "final_test_acc": test_acc,
        "best_test_acc": None if best_test_acc == float("-inf") else best_test_acc,
        "best_epoch": best_epoch,
        "elapsed_s": elapsed,
        "completed_epochs": last_completed_epoch + 1,
        "completed_updates": global_update,
        "timestamp": time.time(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    write_metric(metrics_path, {"event": "final", **summary})
    profiler.write_summary()

    print(f"\nFinal test accuracy: {test_acc:.4f} | Wall-clock: {elapsed:.1f}s")

    return frozen_params, params, noiser_params, test_acc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SNN with EGGROLL")
    parser.add_argument("--dataset", type=str, default=None, choices=["mnist", "cifar10"])
    parser.add_argument("--model_name", type=str, default=None, choices=["mlp_snn", "spiking_resnet18"])
    parser.add_argument("--N", type=int, default=None, help="Override n_inputs")
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--resnet_channels_base", type=int, default=None)
    parser.add_argument("--resnet_norm", type=str, default=None, choices=["group", "batch", "bntt"])
    parser.add_argument("--resnet_norm_groups", type=int, default=None)
    parser.add_argument("--resnet_bn_momentum", type=float, default=None)
    parser.add_argument("--resnet_bn_eps", type=float, default=None)
    parser.add_argument("--resnet_bntt_momentum", type=float, default=None)
    parser.add_argument("--resnet_bntt_eps", type=float, default=None)
    parser.add_argument(
        "--resnet_bntt_affine_bias",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--conv_es_mode",
        type=str,
        default=None,
        choices=["kernel_lora", "matrix_lora"],
    )
    parser.add_argument("--resnet_threshold_scale", action="store_true", help="Scale threshold by 2**stage_idx per stage")
    parser.add_argument(
        "--selective_stage_perturbation",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--stage_perturbation_schedule",
        type=str,
        default=None,
        choices=["head_last_then_last2"],
    )
    parser.add_argument("--stage_perturbation_early_fraction", type=float, default=None)
    parser.add_argument("--stage_perturbation_full_epoch_interval", type=int, default=None)
    parser.add_argument("--pop_size", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--sigma_min", type=float, default=None)
    parser.add_argument("--sigma_max", type=float, default=None)
    parser.add_argument("--sigma_target_success", type=float, default=None)
    parser.add_argument("--sigma_success_tolerance", type=float, default=None)
    parser.add_argument("--sigma_growth", type=float, default=None)
    parser.add_argument("--sigma_decay", type=float, default=None)
    parser.add_argument("--sigma_ema_decay", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fitness_shaping", type=str, default=None, choices=["zscore", "centered_rank"])
    parser.add_argument(
        "--use_batched_update",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use batched EGGROLL parameter updates",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk population eval (0=no chunking)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--updates_per_epoch", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--membrane_readout", action="store_true", help="Use accumulated membrane V as logits")
    parser.add_argument("--escape_noise", action="store_true", help="Stochastic LIF (for SG baseline only)")
    parser.add_argument("--escape_beta", type=float, default=None)
    parser.add_argument("--escape_lambda0", type=float, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "bfloat16"])
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num_test_eval_samples", type=int, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--test_interval", type=int, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=None)
    parser.add_argument("--sigma_warmup_epochs", type=int, default=None)
    parser.add_argument(
        "--profile_mode",
        type=str,
        default=None,
        choices=["off", "startup", "steady_state", "full"],
    )
    parser.add_argument("--profile_trace_dir", type=str, default=None)
    parser.add_argument("--profile_server_port", type=int, default=None)
    parser.add_argument("--profile_max_snapshots", type=int, default=None)
    parser.add_argument("--profile_warmup_updates", type=int, default=None)
    parser.add_argument("--profile_updates_window", type=int, default=None)
    parser.add_argument("--profile_eval_once", action="store_true")
    parser.add_argument(
        "--profile_sync_timings",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def build_config_from_args(args) -> SNNConfig:
    base_cfg = SNNConfig()
    dataset = args.dataset or base_cfg.dataset
    model_name = args.model_name or base_cfg.model_name
    is_cifar_resnet = dataset == "cifar10" and model_name == "spiking_resnet18"

    dataset_defaults = {
        "mnist": {"n_inputs": 784, "in_channels": 1, "image_size": 28},
        "cifar10": {"n_inputs": 3072, "in_channels": 3, "image_size": 32},
    }
    dflt = dataset_defaults[dataset]

    # Precedence: explicit CLI value > dataset-derived default > SNNConfig default.
    return SNNConfig(
        dataset=dataset,
        model_name=model_name,
        n_inputs=args.N if args.N is not None else dflt["n_inputs"],
        hidden_size=args.hidden_size if args.hidden_size is not None else base_cfg.hidden_size,
        resnet_channels_base=(
            args.resnet_channels_base
            if args.resnet_channels_base is not None
            else base_cfg.resnet_channels_base
        ),
        resnet_block_counts=base_cfg.resnet_block_counts,
        resnet_norm=args.resnet_norm or base_cfg.resnet_norm,
        resnet_norm_groups=(
            args.resnet_norm_groups
            if args.resnet_norm_groups is not None
            else base_cfg.resnet_norm_groups
        ),
        resnet_bn_momentum=(
            args.resnet_bn_momentum
            if args.resnet_bn_momentum is not None
            else base_cfg.resnet_bn_momentum
        ),
        resnet_bn_eps=(
            args.resnet_bn_eps
            if args.resnet_bn_eps is not None
            else base_cfg.resnet_bn_eps
        ),
        resnet_bntt_momentum=(
            args.resnet_bntt_momentum
            if args.resnet_bntt_momentum is not None
            else base_cfg.resnet_bntt_momentum
        ),
        resnet_bntt_eps=(
            args.resnet_bntt_eps
            if args.resnet_bntt_eps is not None
            else base_cfg.resnet_bntt_eps
        ),
        resnet_bntt_affine_bias=(
            args.resnet_bntt_affine_bias
            if args.resnet_bntt_affine_bias is not None
            else base_cfg.resnet_bntt_affine_bias
        ),
        conv_es_mode=(
            args.conv_es_mode
            if args.conv_es_mode is not None
            else base_cfg.conv_es_mode
        ),
        resnet_threshold_scale=args.resnet_threshold_scale,
        selective_stage_perturbation=(
            args.selective_stage_perturbation
            if args.selective_stage_perturbation is not None
            else base_cfg.selective_stage_perturbation
        ),
        stage_perturbation_schedule=(
            args.stage_perturbation_schedule
            if args.stage_perturbation_schedule is not None
            else base_cfg.stage_perturbation_schedule
        ),
        stage_perturbation_early_fraction=(
            args.stage_perturbation_early_fraction
            if args.stage_perturbation_early_fraction is not None
            else base_cfg.stage_perturbation_early_fraction
        ),
        stage_perturbation_full_epoch_interval=(
            args.stage_perturbation_full_epoch_interval
            if args.stage_perturbation_full_epoch_interval is not None
            else base_cfg.stage_perturbation_full_epoch_interval
        ),
        n_classes=base_cfg.n_classes,
        timesteps=args.timesteps if args.timesteps is not None else base_cfg.timesteps,
        pop_size=args.pop_size if args.pop_size is not None else base_cfg.pop_size,
        rank=args.rank if args.rank is not None else base_cfg.rank,
        sigma=args.sigma if args.sigma is not None else base_cfg.sigma,
        sigma_min=args.sigma_min if args.sigma_min is not None else base_cfg.sigma_min,
        sigma_max=(
            args.sigma_max
            if args.sigma_max is not None
            else (0.012 if is_cifar_resnet else base_cfg.sigma_max)
        ),
        sigma_target_success=(
            args.sigma_target_success
            if args.sigma_target_success is not None
            else base_cfg.sigma_target_success
        ),
        sigma_success_tolerance=(
            args.sigma_success_tolerance
            if args.sigma_success_tolerance is not None
            else base_cfg.sigma_success_tolerance
        ),
        sigma_growth=(
            args.sigma_growth
            if args.sigma_growth is not None
            else base_cfg.sigma_growth
        ),
        sigma_decay=(
            args.sigma_decay
            if args.sigma_decay is not None
            else base_cfg.sigma_decay
        ),
        sigma_ema_decay=(
            args.sigma_ema_decay
            if args.sigma_ema_decay is not None
            else base_cfg.sigma_ema_decay
        ),
        lr=args.lr if args.lr is not None else base_cfg.lr,
        fitness_shaping=(
            args.fitness_shaping
            if args.fitness_shaping is not None
            else ("centered_rank" if is_cifar_resnet else base_cfg.fitness_shaping)
        ),
        use_batched_update=(
            args.use_batched_update
            if args.use_batched_update is not None
            else (True if is_cifar_resnet else base_cfg.use_batched_update)
        ),
        batch_size=args.batch_size if args.batch_size is not None else base_cfg.batch_size,
        num_epochs=args.epochs if args.epochs is not None else base_cfg.num_epochs,
        updates_per_epoch=(
            args.updates_per_epoch
            if args.updates_per_epoch is not None
            else base_cfg.updates_per_epoch
        ),
        chunk_size=args.chunk_size if args.chunk_size is not None else base_cfg.chunk_size,
        threshold=args.threshold if args.threshold is not None else base_cfg.threshold,
        membrane_readout=args.membrane_readout,
        escape_noise=args.escape_noise,
        escape_beta=args.escape_beta if args.escape_beta is not None else base_cfg.escape_beta,
        escape_lambda0=args.escape_lambda0 if args.escape_lambda0 is not None else base_cfg.escape_lambda0,
        seed=args.seed if args.seed is not None else base_cfg.seed,
        data_path=args.data_path or base_cfg.data_path,
        dtype=args.dtype or base_cfg.dtype,
        in_channels=args.in_channels if args.in_channels is not None else dflt["in_channels"],
        image_size=args.image_size if args.image_size is not None else dflt["image_size"],
        augment=args.augment,
        num_test_eval_samples=(
            args.num_test_eval_samples
            if args.num_test_eval_samples is not None
            else base_cfg.num_test_eval_samples
        ),
        run_name=args.run_name or base_cfg.run_name,
        log_dir=args.log_dir or base_cfg.log_dir,
        checkpoint_dir=args.checkpoint_dir or base_cfg.checkpoint_dir,
        resume_from=args.resume_from,
        log_interval=args.log_interval if args.log_interval is not None else base_cfg.log_interval,
        test_interval=args.test_interval if args.test_interval is not None else base_cfg.test_interval,
        checkpoint_interval=(
            args.checkpoint_interval
            if args.checkpoint_interval is not None
            else base_cfg.checkpoint_interval
        ),
        sigma_warmup_epochs=(
            args.sigma_warmup_epochs
            if args.sigma_warmup_epochs is not None
            else base_cfg.sigma_warmup_epochs
        ),
        profile_mode=args.profile_mode or base_cfg.profile_mode,
        profile_trace_dir=args.profile_trace_dir or base_cfg.profile_trace_dir,
        profile_server_port=(
            args.profile_server_port
            if args.profile_server_port is not None
            else base_cfg.profile_server_port
        ),
        profile_max_snapshots=(
            args.profile_max_snapshots
            if args.profile_max_snapshots is not None
            else base_cfg.profile_max_snapshots
        ),
        profile_warmup_updates=(
            args.profile_warmup_updates
            if args.profile_warmup_updates is not None
            else base_cfg.profile_warmup_updates
        ),
        profile_updates_window=(
            args.profile_updates_window
            if args.profile_updates_window is not None
            else base_cfg.profile_updates_window
        ),
        profile_eval_once=args.profile_eval_once,
        profile_sync_timings=(
            args.profile_sync_timings
            if args.profile_sync_timings is not None
            else base_cfg.profile_sync_timings
        ),
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
