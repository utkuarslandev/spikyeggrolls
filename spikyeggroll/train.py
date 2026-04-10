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
from spikyeggroll.models.snn import SNNModel
from spikyeggroll.data.mnist import load_mnist, encode_batch


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

    # Initialize model
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(model_key, cfg)
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
    train_data, train_labels, test_data, test_labels = load_mnist(
        cfg.data_path + "/mnist"
    )
    _timesteps = cfg.timesteps
    _batch_size = cfg.batch_size

    @jax.jit
    def sample_and_encode(images, labels, key):
        k1, k2 = jax.random.split(key)
        indices = jax.random.choice(k1, images.shape[0], shape=(_batch_size,), replace=False)
        batch_imgs = images[indices]
        probs = batch_imgs[:, None, :]
        uniform = jax.random.uniform(k2, (batch_imgs.shape[0], _timesteps, batch_imgs.shape[1]))
        spikes = (uniform < probs).astype(jnp.float32)
        return spikes, labels[indices]

    print(f"Dataset: {cfg.dataset} | train: {train_data.shape}, test: {test_data.shape}")
    print(f"Architecture: {cfg.n_inputs}-{cfg.hidden_size}-{cfg.hidden_size}-{cfg.n_classes}")
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
            lambda n, p, i, x, l1b: SNNModel.forward(
                EggRoll, frozen_noiser_params, n,
                frozen_params, p, es_tree_key, i, x, l1b
            ),
            in_axes=(None, None, 0, None, None),
        )
    )

    # JIT-compiled forward: evaluation (no noise, iterinfo=None)
    jit_forward_eval = jax.jit(
        lambda n, p, x: SNNModel.forward(
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
    n_test_chunks = test_data.shape[0] // cfg.batch_size
    def eval_test():
        all_preds = []
        for i in range(n_test_chunks):
            start = i * cfg.batch_size
            te_chunk = test_data[start:start+cfg.batch_size]
            te_key = jax.random.fold_in(jax.random.key(999), i)
            te_spikes = encode_batch(te_chunk, cfg.timesteps, te_key)
            out = jit_forward_eval(noiser_params, params, te_spikes)
            all_preds.append(jnp.argmax(out, axis=-1))
        return float(jnp.mean(jnp.concatenate(all_preds) == test_labels[:n_test_chunks * cfg.batch_size]))

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

            # Precompute layer 1 base matmul (shared across population)
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

            # Update parameters
            noiser_params, params = jit_update(noiser_params, params, fitnesses, iterinfo)

            # 1/5th success rule: adapt sigma based on fraction beating baseline
            n_better = int(jnp.sum(raw_scores > val_fitness))
            success_rate = n_better / N

            if ema_success is None:
                ema_success = success_rate
            else:
                ema_success = 0.9 * ema_success + 0.1 * success_rate

            if ema_success > 0.2:
                noiser_params["sigma"] = noiser_params["sigma"] * 1.02
            elif ema_success < 0.2:
                noiser_params["sigma"] = noiser_params["sigma"] / 1.02

            do_log = cfg.log_interval > 0 and epoch % cfg.log_interval == 0
            do_test = cfg.test_interval > 0 and epoch % cfg.test_interval == 0
            do_checkpoint = cfg.checkpoint_interval > 0 and epoch % cfg.checkpoint_interval == 0

            val_acc = float(jnp.mean(jnp.argmax(val_out, axis=-1) == y_batch))
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
                "n_better": n_better,
                "pop_size": N,
                "success_rate": success_rate,
                "ema_success": float(ema_success),
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
                    f"better: {n_better:4d}/{N} ema:{float(ema_success):.3f} | "
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


def main():
    parser = argparse.ArgumentParser(description="Train SNN with EGGROLL")
    parser.add_argument("--N", type=int, default=None, help="Override n_inputs")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--pop_size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--chunk_size", type=int, default=0, help="Chunk population eval (0=no chunking)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--membrane_readout", action="store_true", help="Use accumulated membrane V as logits")
    parser.add_argument("--escape_noise", action="store_true", help="Stochastic LIF (for SG baseline only)")
    parser.add_argument("--escape_beta", type=float, default=50.0)
    parser.add_argument("--escape_lambda0", type=float, default=1.0)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--log_dir", type=str, default="logs/spikyeggroll")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/spikyeggroll")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--test_interval", type=int, default=100)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    args = parser.parse_args()

    n_inputs = args.N or 784
    n_classes = 10
    timesteps = args.timesteps or 25

    cfg = SNNConfig(
        n_inputs=n_inputs,
        hidden_size=args.hidden_size,
        n_classes=n_classes,
        timesteps=timesteps,
        pop_size=args.pop_size,
        rank=args.rank,
        sigma=args.sigma,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        chunk_size=args.chunk_size,
        threshold=args.threshold,
        membrane_readout=args.membrane_readout,
        escape_noise=args.escape_noise,
        escape_beta=args.escape_beta,
        escape_lambda0=args.escape_lambda0,
        seed=args.seed,
        data_path=args.data_path,
        run_name=args.run_name,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        log_interval=args.log_interval,
        test_interval=args.test_interval,
        checkpoint_interval=args.checkpoint_interval,
    )

    train(cfg)


if __name__ == "__main__":
    main()
