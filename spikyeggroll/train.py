"""Main training loop using EGGROLL evolution strategies."""

import argparse
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
from spikyeggroll.runtime import get_dataset_spec, get_model_cls


def _tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def _tree_to_jax(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


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
        "data_key": np.asarray(jax.random.key_data(data_key)),
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
    payload["data_key"] = jax.random.wrap_key_data(jnp.asarray(payload["data_key"]))
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


def summarize_output_activity(outputs):
    return {
        "output_nonzero_fraction": float(jnp.mean(outputs != 0)),
        "output_class_variance_mean": float(jnp.mean(jnp.var(outputs, axis=-1))),
    }


def train(cfg: SNNConfig = None):
    """Run EGGROLL training.

    Args:
        cfg: hyperparameter config, uses defaults if None

    Returns:
        (frozen_params, params, noiser_params, test_accuracy)
    """
    if cfg is None:
        cfg = SNNConfig()

    log_dir = Path(cfg.log_dir)
    checkpoint_dir = Path(cfg.checkpoint_dir)
    metrics_path = log_dir / f"{cfg.run_name}.metrics.jsonl"
    summary_path = log_dir / f"{cfg.run_name}.summary.json"

    N = cfg.pop_size
    assert N % 2 == 0, "Population size must be even (antithetical sampling)"

    # PRNG keys
    key = jax.random.key(cfg.seed)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    data_key = jax.random.fold_in(key, 2)

    model_cls = get_model_cls(cfg.model_name)

    # Initialize model
    frozen_params, params, scan_map, es_map = model_cls.rand_init(model_key, cfg)
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)

    # Initialize EGGROLL noiser
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        cfg.sigma,
        cfg.lr,
        solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999},
        rank=cfg.rank,
    )

    start_epoch = 0
    ema_success = None
    best_test_acc = float("-inf")
    best_epoch = None

    if cfg.resume_from:
        checkpoint = load_checkpoint(cfg.resume_from)
        params = checkpoint["params"]
        noiser_params = checkpoint["noiser_params"]
        data_key = checkpoint["data_key"]
        start_epoch = int(checkpoint["epoch"]) + 1
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
                "timestamp": time.time(),
            },
        )

    # Load dataset
    dataset_spec = get_dataset_spec(cfg)
    train_data, train_labels, test_data, test_labels = dataset_spec.loader()
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

    @jax.jit
    def sample_and_encode(images, labels, key):
        k1, k2 = jax.random.split(key)
        indices = jax.random.choice(k1, images.shape[0], shape=(_batch_size,), replace=False)
        batch_imgs = images[indices]
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
            f"classes={cfg.n_classes}"
        )
    print(f"EGGROLL: pop={N}, rank={cfg.rank}, sigma={cfg.sigma}, lr={cfg.lr}")
    print(f"Run: {cfg.run_name} | metrics: {metrics_path} | checkpoints: {checkpoint_dir}")

    # Precompute layer 1 base: x @ W1.T for all timesteps (shared across population)
    @jax.jit
    def compute_l1_base(params, x):
        """Compute x @ W1.T once, returns [T, B, hidden]."""
        x_t = jnp.transpose(x, (1, 0, 2))  # [T, B, n_inputs]
        W1 = params["linear1"]["weight"]     # [hidden, n_inputs]
        return x_t @ W1.T                    # [T, B, hidden]

    # JIT-compiled forward: population evaluation (with noise)
    # vmap over iterinfo only — input batch and l1_base shared across population
    jit_forward = jax.jit(
        jax.vmap(
            lambda n, p, i, x, l1b: model_cls.forward(
                EggRoll, frozen_noiser_params, n,
                frozen_params, p, es_tree_key, i, x, l1b
            ),
            in_axes=(None, None, 0, None, None),
        )
    )

    # JIT-compiled forward: evaluation (no noise, iterinfo=None)
    jit_forward_eval = jax.jit(
        lambda n, p, x: model_cls.forward(
            EggRoll, frozen_noiser_params, n,
            frozen_params, p, es_tree_key, None, x
        )
    )

    # JIT-compiled parameter update
    jit_update = jax.jit(
        lambda n, p, f, i: EggRoll.do_updates(
            frozen_noiser_params, n, p, es_tree_key, f, i, es_map
        )
    )

    # Evaluate test set in fixed-size chunks (same batch size as training).
    # Note: remainder samples are intentionally dropped to avoid shape-triggered recompiles.
    eval_test_size = test_data.shape[0]
    if cfg.num_test_eval_samples and cfg.num_test_eval_samples > 0:
        eval_test_size = min(eval_test_size, cfg.num_test_eval_samples)
    n_test_chunks = eval_test_size // cfg.batch_size

    def eval_test():
        if n_test_chunks == 0:
            return float("nan")
        all_preds = []
        for i in range(n_test_chunks):
            start = i * cfg.batch_size
            te_chunk = test_data[start:start+cfg.batch_size]
            te_key = jax.random.fold_in(jax.random.key(999), i)
            te_spikes = encode_batch_fn(te_chunk, cfg.timesteps, te_key)
            out = jit_forward_eval(noiser_params, params, te_spikes)
            all_preds.append(jnp.argmax(out, axis=-1))
        end = n_test_chunks * cfg.batch_size
        return float(jnp.mean(jnp.concatenate(all_preds) == test_labels[:end]))

    write_metric(
        metrics_path,
        {
            "event": "start",
            "run_name": cfg.run_name,
            "cfg": asdict(cfg),
            "timestamp": time.time(),
            "start_epoch": start_epoch,
        },
    )

    # Training loop
    t_start = time.time()
    last_completed_epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, cfg.num_epochs):
            data_key, batch_key, _encode_key = jax.random.split(data_key, 3)

            # Sample mini-batch with Poisson encoding (same for all population members)
            x_batch, y_batch = sample_and_encode(train_data, train_labels, batch_key)

            # Build iterinfo for population
            iterinfo = (jnp.full(N, epoch, dtype=jnp.int32), jnp.arange(N))

            # Evaluate base params (no noise) on this batch
            val_out = jit_forward_eval(noiser_params, params, x_batch)
            val_fitness = compute_fitness(val_out, y_batch)

            # Precompute layer 1 base matmul when model exposes linear1
            l1_base = None
            if "linear1" in params:
                l1_base = compute_l1_base(params, x_batch)

            # Evaluate population (with noise) — chunked to fit in GPU memory
            if cfg.chunk_size > 0 and cfg.chunk_size < N:
                score_chunks = []
                for c_start in range(0, N, cfg.chunk_size):
                    c_end = min(c_start + cfg.chunk_size, N)
                    c_iter = (iterinfo[0][c_start:c_end], iterinfo[1][c_start:c_end])
                    c_out = jit_forward(noiser_params, params, c_iter, x_batch, l1_base)
                    c_scores = jax.vmap(compute_fitness, in_axes=(0, None))(c_out, y_batch)
                    score_chunks.append(c_scores)
                raw_scores = jnp.concatenate(score_chunks)
            else:
                pop_out = jit_forward(noiser_params, params, iterinfo, x_batch, l1_base)
                raw_scores = jax.vmap(compute_fitness, in_axes=(0, None))(pop_out, y_batch)

            # Z-score fitness shaping
            fitnesses = EggRoll.convert_fitnesses(
                frozen_noiser_params, noiser_params, raw_scores
            )
            raw_score_std = float(jnp.std(raw_scores))

            # Update parameters
            noiser_params, params = jit_update(noiser_params, params, fitnesses, iterinfo)

            # 1/5th success rule: adapt sigma based on fraction beating baseline
            n_better = int(jnp.sum(raw_scores > val_fitness))
            success_rate = n_better / N

            if ema_success is None:
                ema_success = success_rate
            else:
                ema_success = 0.9 * ema_success + 0.1 * success_rate

            if epoch >= cfg.sigma_warmup_epochs:
                if ema_success > 0.2:
                    noiser_params["sigma"] = noiser_params["sigma"] * 1.02
                elif ema_success < 0.2:
                    noiser_params["sigma"] = noiser_params["sigma"] / 1.02

            noiser_params["sigma"] = jnp.maximum(noiser_params["sigma"], cfg.sigma_min)

            do_log = cfg.log_interval > 0 and epoch % cfg.log_interval == 0
            do_test = cfg.test_interval > 0 and epoch % cfg.test_interval == 0
            do_checkpoint = cfg.checkpoint_interval > 0 and epoch % cfg.checkpoint_interval == 0

            val_acc = float(jnp.mean(jnp.argmax(val_out, axis=-1) == y_batch))
            output_activity = summarize_output_activity(val_out)
            elapsed = time.time() - t_start
            eps = (epoch - start_epoch + 1) / elapsed if elapsed > 0 else 0.0
            test_acc = eval_test() if do_test else None

            record = {
                "event": "epoch",
                "epoch": epoch,
                "elapsed_s": elapsed,
                "epochs_per_s": eps,
                "val_fitness": float(val_fitness),
                "val_acc": val_acc,
                "sigma": float(noiser_params["sigma"]),
                "raw_score_std": raw_score_std,
                "n_better": n_better,
                "pop_size": N,
                "success_rate": success_rate,
                "ema_success": float(ema_success),
                **output_activity,
                "test_acc": test_acc,
                "timestamp": time.time(),
            }
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
                    f"{eps:.1f} ep/s"
                    f"{test_str}"
                )
            last_completed_epoch = epoch
    except KeyboardInterrupt:
        interrupt_path = save_checkpoint(
            checkpoint_dir,
            cfg.run_name,
            "interrupt",
            cfg,
            last_completed_epoch,
            params,
            noiser_params,
            data_key,
            ema_success,
            best_test_acc,
            best_epoch,
        )
        print(f"\nInterrupted. Saved checkpoint: {interrupt_path}")
        raise

    # Final test accuracy
    test_acc = eval_test()
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
        "timestamp": time.time(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    write_metric(metrics_path, {"event": "final", **summary})

    print(f"\nFinal test accuracy: {test_acc:.4f} | Wall-clock: {elapsed:.1f}s")

    return frozen_params, params, noiser_params, test_acc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SNN with EGGROLL")
    parser.add_argument("--dataset", type=str, default=None, choices=["mnist", "cifar10"])
    parser.add_argument("--model_name", type=str, default=None, choices=["mlp_snn", "spiking_resnet18"])
    parser.add_argument("--N", type=int, default=None, help="Override n_inputs")
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--resnet_channels_base", type=int, default=None)
    parser.add_argument("--resnet_norm", type=str, default=None, choices=["group"])
    parser.add_argument("--resnet_norm_groups", type=int, default=None)
    parser.add_argument("--pop_size", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--sigma_min", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None, help="Chunk population eval (0=no chunking)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--membrane_readout", action="store_true", help="Use accumulated membrane V as logits")
    parser.add_argument("--escape_noise", action="store_true", help="Stochastic LIF (for SG baseline only)")
    parser.add_argument("--escape_beta", type=float, default=None)
    parser.add_argument("--escape_lambda0", type=float, default=None)
    parser.add_argument("--data_path", type=str, default=None)
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
    return parser


def build_config_from_args(args) -> SNNConfig:
    base_cfg = SNNConfig()
    dataset = args.dataset or base_cfg.dataset

    dataset_defaults = {
        "mnist": {"n_inputs": 784, "in_channels": 1, "image_size": 28},
        "cifar10": {"n_inputs": 3072, "in_channels": 3, "image_size": 32},
    }
    dflt = dataset_defaults[dataset]

    # Precedence: explicit CLI value > dataset-derived default > SNNConfig default.
    return SNNConfig(
        dataset=dataset,
        model_name=args.model_name or base_cfg.model_name,
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
        n_classes=base_cfg.n_classes,
        timesteps=args.timesteps if args.timesteps is not None else base_cfg.timesteps,
        pop_size=args.pop_size if args.pop_size is not None else base_cfg.pop_size,
        rank=args.rank if args.rank is not None else base_cfg.rank,
        sigma=args.sigma if args.sigma is not None else base_cfg.sigma,
        sigma_min=args.sigma_min if args.sigma_min is not None else base_cfg.sigma_min,
        lr=args.lr if args.lr is not None else base_cfg.lr,
        batch_size=args.batch_size if args.batch_size is not None else base_cfg.batch_size,
        num_epochs=args.epochs if args.epochs is not None else base_cfg.num_epochs,
        chunk_size=args.chunk_size if args.chunk_size is not None else base_cfg.chunk_size,
        threshold=args.threshold if args.threshold is not None else base_cfg.threshold,
        membrane_readout=args.membrane_readout,
        escape_noise=args.escape_noise,
        escape_beta=args.escape_beta if args.escape_beta is not None else base_cfg.escape_beta,
        escape_lambda0=args.escape_lambda0 if args.escape_lambda0 is not None else base_cfg.escape_lambda0,
        seed=args.seed if args.seed is not None else base_cfg.seed,
        data_path=args.data_path or base_cfg.data_path,
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
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    cfg = build_config_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
