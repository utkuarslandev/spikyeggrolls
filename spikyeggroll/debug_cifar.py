"""Debug probe for CIFAR-10 spiking ResNet training collapse."""

import argparse
import json

import jax
import jax.numpy as jnp

from hyperscalees.models.base_model import CommonParams
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.noiser.eggroll import EggRoll

from spikyeggroll.configs import SNNConfig
from spikyeggroll.data.cifar10 import encode_batch, load_cifar10
from spikyeggroll.models.spiking_resnet import SpikingResNet18Model
from spikyeggroll.train import compute_fitness


def _make_cfg(args) -> SNNConfig:
    return SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        n_inputs=3072,
        n_classes=10,
        timesteps=args.timesteps,
        pop_size=args.pop_size,
        rank=args.rank,
        sigma=args.sigma,
        lr=args.lr,
        batch_size=args.batch_size,
        data_path=args.data_path,
        resnet_width=args.resnet_width,
        resnet_blocks=args.resnet_blocks,
        membrane_readout=args.membrane_readout,
    )


def _load_batch(cfg: SNNConfig, batch_size: int, seed: int, use_real_data: bool):
    key = jax.random.key(seed)
    if use_real_data:
        train_images, train_labels, _, _ = load_cifar10(cfg.data_path + "/cifar10")
        idx = jax.random.choice(key, train_images.shape[0], shape=(batch_size,), replace=False)
        images = train_images[idx]
        labels = train_labels[idx]
    else:
        image_key, label_key = jax.random.split(key)
        images = jax.random.uniform(image_key, (batch_size, 32, 32, 3))
        labels = jax.random.randint(label_key, (batch_size,), 0, cfg.n_classes)
    encode_key = jax.random.fold_in(key, 1)
    spikes = encode_batch(images, cfg.timesteps, encode_key)
    return spikes, labels


def main():
    parser = argparse.ArgumentParser(description="Probe CIFAR spiking ResNet activity at init.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--resnet_width", type=int, default=768)
    parser.add_argument("--resnet_blocks", type=int, default=8)
    parser.add_argument("--pop_size", type=int, default=512)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--real_data", action="store_true", help="Use a real CIFAR-10 batch instead of synthetic inputs.")
    parser.add_argument("--membrane_readout", action="store_true")
    args = parser.parse_args()

    cfg = _make_cfg(args)
    spikes, labels = _load_batch(cfg, args.batch_size, args.seed, args.real_data)

    model_key, es_key = jax.random.split(jax.random.key(args.seed))
    frozen_params, params, scan_map, _ = SpikingResNet18Model.rand_init(model_key, cfg)
    es_tree_key = simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params,
        cfg.sigma,
        cfg.lr,
        rank=cfg.rank,
    )

    common_params = CommonParams(
        EggRoll,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        None,
    )

    outputs, activity_stats = SpikingResNet18Model.forward_debug(common_params, spikes)
    base_fitness = float(compute_fitness(outputs, labels))
    predictions = jnp.argmax(outputs, axis=-1)
    base_acc = float(jnp.mean(predictions == labels))

    iterinfo = (jnp.full(cfg.pop_size, 0, dtype=jnp.int32), jnp.arange(cfg.pop_size))
    pop_out = jax.vmap(
        lambda n, p, i, x: SpikingResNet18Model.forward(
            EggRoll,
            frozen_noiser_params,
            n,
            frozen_params,
            p,
            es_tree_key,
            i,
            x,
        ),
        in_axes=(None, None, 0, None),
    )(noiser_params, params, iterinfo, spikes)
    raw_scores = jax.vmap(compute_fitness, in_axes=(0, None))(pop_out, labels)

    result = {
        "cfg": {
            "timesteps": cfg.timesteps,
            "resnet_width": cfg.resnet_width,
            "resnet_blocks": cfg.resnet_blocks,
            "pop_size": cfg.pop_size,
            "rank": cfg.rank,
            "sigma": cfg.sigma,
            "lr": cfg.lr,
            "membrane_readout": cfg.membrane_readout,
            "real_data": args.real_data,
        },
        "base_metrics": {
            "fitness": base_fitness,
            "accuracy": base_acc,
            "output_shape": list(outputs.shape),
            "predicted_classes": [int(v) for v in predictions[:16]],
        },
        "population_metrics": {
            "raw_score_mean": float(jnp.mean(raw_scores)),
            "raw_score_std": float(jnp.std(raw_scores)),
            "raw_score_min": float(jnp.min(raw_scores)),
            "raw_score_max": float(jnp.max(raw_scores)),
            "n_better_than_base": int(jnp.sum(raw_scores > base_fitness)),
            "population_output_variance_mean": float(jnp.mean(jnp.var(pop_out, axis=0))),
        },
        "activity": activity_stats,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
