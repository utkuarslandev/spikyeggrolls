"""Tests for SNNModel initialization and forward pass."""

import pytest
import jax
import jax.numpy as jnp
import json
from pathlib import Path

from spikyeggroll.configs import SNNConfig
from spikyeggroll.eval import evaluate
from spikyeggroll.train import build_config_from_args, build_parser, train
from spikyeggroll.models.snn import SNNModel
from spikyeggroll.models.spiking_resnet import SpikingResNet18Model
from hyperscalees.models.common import simple_es_tree_key
from hyperscalees.noiser.eggroll import EggRoll
from spikyeggroll.runtime import DatasetSpec


def test_rand_init_structure():
    """rand_init should return a valid CommonInit with expected keys."""
    key = jax.random.key(42)
    cfg = SNNConfig(hidden_size=32, n_inputs=16, n_classes=5)
    frozen_params, params, scan_map, es_map = SNNModel.rand_init(key, cfg)

    assert "linear1" in params
    assert "linear2" in params
    assert "linear_out" in params
    assert frozen_params["beta"] == cfg.beta
    assert frozen_params["threshold"] == cfg.threshold


def test_rand_init_shapes():
    """Weight matrices should have correct shapes."""
    key = jax.random.key(42)
    cfg = SNNConfig(hidden_size=32, n_inputs=16, n_classes=5)
    _, params, _, _ = SNNModel.rand_init(key, cfg)

    # MM stores weights as (out_dim, in_dim)
    assert params["linear1"]["weight"].shape == (32, 16)
    assert params["linear2"]["weight"].shape == (32, 32)
    assert params["linear_out"]["weight"].shape == (5, 32)


def test_spiking_resnet_init_structure():
    """SpikingResNet18 init should expose conv stem/output and residual stage params."""
    key = jax.random.key(7)
    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        n_inputs=3072,
        n_classes=10,
    )
    frozen_params, params, _, _ = SpikingResNet18Model.rand_init(key, cfg)
    assert "stem_conv" in params
    assert "stem_norm" in params
    assert "linear_out" in params
    assert "stage0_block0" in params
    assert "stage3_block1" in params
    assert params["stem_conv"]["weight"].shape == (64, 3, 3, 3)
    assert params["linear_out"]["weight"].shape == (10, 512)
    assert tuple(frozen_params["stage_blocks"]) == (2, 2, 2, 2)


@pytest.mark.parametrize(
    ("cfg", "model_cls", "input_shape"),
    [
        (
            SNNConfig(
                dataset="mnist",
                model_name="mlp_snn",
                n_inputs=784,
                hidden_size=32,
                n_classes=10,
                timesteps=5,
                pop_size=8,
            ),
            SNNModel,
            (4, 5, 784),
        ),
        (
            SNNConfig(
                dataset="cifar10",
                model_name="spiking_resnet18",
                n_inputs=3072,
                n_classes=10,
                timesteps=4,
                pop_size=8,
            ),
            SpikingResNet18Model,
            (4, 4, 3, 32, 32),
        ),
    ],
)
def test_evaluate_supports_all_models(cfg, model_cls, input_shape):
    key = jax.random.key(123)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    frozen_params, params, scan_map, _ = model_cls.rand_init(k1, cfg)
    es_tree_key = simple_es_tree_key(params, k2, scan_map)
    frozen_noiser_params, noiser_params = EggRoll.init_noiser(
        params, cfg.sigma, cfg.lr, rank=cfg.rank
    )

    test_data = jax.random.bernoulli(k3, 0.25, input_shape).astype(jnp.float32)
    test_labels = jax.random.randint(k4, (input_shape[0],), 0, cfg.n_classes)

    acc = evaluate(
        cfg,
        frozen_noiser_params,
        noiser_params,
        frozen_params,
        params,
        es_tree_key,
        test_data,
        test_labels,
    )

    assert 0.0 <= acc <= 1.0


def test_cli_defaults_match_snnconfig():
    parser = build_parser()
    args = parser.parse_args([])

    cfg = build_config_from_args(args)
    expected = SNNConfig()

    assert cfg == expected


def test_cli_explicit_override_only_changes_target_field():
    parser = build_parser()
    args = parser.parse_args(["--pop_size", "256"])

    cfg = build_config_from_args(args)
    expected = SNNConfig(pop_size=256)

    assert cfg == expected


def test_cli_dataset_defaults_switch_with_cifar10():
    parser = build_parser()
    args = parser.parse_args(["--dataset", "cifar10"])

    cfg = build_config_from_args(args)

    assert cfg.dataset == "cifar10"
    assert cfg.n_inputs == 3072
    assert cfg.in_channels == 3
    assert cfg.image_size == 32
    assert cfg.pop_size == SNNConfig().pop_size


def test_cli_cifar_resnet_defaults_enable_perf_path():
    parser = build_parser()
    args = parser.parse_args(["--dataset", "cifar10", "--model_name", "spiking_resnet18"])

    cfg = build_config_from_args(args)

    assert cfg.fitness_shaping == "centered_rank"
    assert cfg.use_batched_update is True
    assert cfg.sigma_min == pytest.approx(0.0025)
    assert cfg.sigma_max == pytest.approx(0.012)


def test_cli_profile_flags_roundtrip():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--profile_mode",
            "full",
            "--profile_trace_dir",
            "/tmp/traces",
            "--profile_server_port",
            "9999",
            "--profile_max_snapshots",
            "7",
            "--profile_warmup_updates",
            "2",
            "--profile_updates_window",
            "4",
            "--profile_eval_once",
            "--no-profile_sync_timings",
            "--resnet_norm",
            "batch",
            "--resnet_bn_momentum",
            "0.8",
            "--resnet_bn_eps",
            "0.0001",
            "--sigma_target_success",
            "0.25",
            "--sigma_success_tolerance",
            "0.05",
            "--sigma_growth",
            "1.03",
            "--sigma_decay",
            "0.98",
            "--sigma_ema_decay",
            "0.85",
        ]
    )

    cfg = build_config_from_args(args)

    assert cfg.profile_mode == "full"
    assert cfg.profile_trace_dir == "/tmp/traces"
    assert cfg.profile_server_port == 9999
    assert cfg.profile_max_snapshots == 7
    assert cfg.profile_warmup_updates == 2
    assert cfg.profile_updates_window == 4
    assert cfg.profile_eval_once is True
    assert cfg.profile_sync_timings is False
    assert cfg.resnet_norm == "batch"
    assert cfg.resnet_bn_momentum == pytest.approx(0.8)
    assert cfg.resnet_bn_eps == pytest.approx(0.0001)
    assert cfg.sigma_target_success == pytest.approx(0.25)
    assert cfg.sigma_success_tolerance == pytest.approx(0.05)
    assert cfg.sigma_growth == pytest.approx(1.03)
    assert cfg.sigma_decay == pytest.approx(0.98)
    assert cfg.sigma_ema_decay == pytest.approx(0.85)


def test_cli_selective_stage_perturbation_flags_roundtrip():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--selective_stage_perturbation",
            "--stage_perturbation_schedule",
            "head_last_then_last2",
            "--stage_perturbation_early_fraction",
            "0.4",
            "--stage_perturbation_full_epoch_interval",
            "6",
        ]
    )

    cfg = build_config_from_args(args)

    assert cfg.selective_stage_perturbation is True
    assert cfg.stage_perturbation_schedule == "head_last_then_last2"
    assert cfg.stage_perturbation_early_fraction == pytest.approx(0.4)
    assert cfg.stage_perturbation_full_epoch_interval == 6


def test_cli_rejects_removed_legacy_resnet_flags():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--resnet_width", "128"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--resnet_blocks", "2"])


def test_train_updates_per_epoch_and_sigma_max(monkeypatch, tmp_path):
    def loader():
        train_images = jnp.linspace(0.0, 1.0, 16 * 8, dtype=jnp.float32).reshape(16, 8)
        train_labels = (jnp.arange(16) % 2).astype(jnp.int32)
        test_images = train_images[:8]
        test_labels = train_labels[:8]
        return train_images, train_labels, test_images, test_labels

    def encoder(images, timesteps, key):
        del key
        return jnp.broadcast_to(images[:, None, :], (images.shape[0], timesteps, images.shape[1]))

    monkeypatch.setattr(
        "spikyeggroll.train.get_dataset_spec",
        lambda cfg: DatasetSpec(
            loader=loader,
            encoder=encoder,
            n_inputs=8,
            in_channels=1,
            image_size=1,
        ),
    )

    cfg = SNNConfig(
        dataset="mnist",
        model_name="mlp_snn",
        n_inputs=8,
        hidden_size=4,
        n_classes=2,
        timesteps=2,
        pop_size=4,
        rank=1,
        sigma=0.02,
        sigma_min=0.001,
        sigma_max=0.005,
        lr=0.001,
        batch_size=2,
        chunk_size=2,
        num_epochs=2,
        updates_per_epoch=3,
        sigma_warmup_epochs=0,
        log_interval=1,
        test_interval=0,
        checkpoint_interval=0,
        run_name="pytest-train-updates",
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
    )

    _, _, _, test_acc = train(cfg)

    metrics_path = tmp_path / "logs" / "pytest-train-updates.metrics.jsonl"
    summary_path = tmp_path / "logs" / "pytest-train-updates.summary.json"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    epoch_records = [r for r in records if r["event"] == "epoch"]
    summary = json.loads(summary_path.read_text())

    assert len(epoch_records) == 2
    assert epoch_records[0]["global_update"] == 3
    assert epoch_records[1]["global_update"] == 6
    assert all(r["sigma"] <= cfg.sigma_max + 1e-8 for r in epoch_records)
    assert all(r["sigma_action"] in {"grow", "hold", "decay"} for r in epoch_records)
    assert summary["completed_updates"] == 6
    assert 0.0 <= test_acc <= 1.0


def _install_dummy_dataset(monkeypatch):
    def loader():
        train_images = jnp.linspace(0.0, 1.0, 16 * 8, dtype=jnp.float32).reshape(16, 8)
        train_labels = (jnp.arange(16) % 2).astype(jnp.int32)
        test_images = train_images[:8]
        test_labels = train_labels[:8]
        return train_images, train_labels, test_images, test_labels

    def encoder(images, timesteps, key):
        del key
        return jnp.broadcast_to(images[:, None, :], (images.shape[0], timesteps, images.shape[1]))

    monkeypatch.setattr(
        "spikyeggroll.train.get_dataset_spec",
        lambda cfg: DatasetSpec(
            loader=loader,
            encoder=encoder,
            n_inputs=8,
            in_channels=1,
            image_size=1,
        ),
    )


def _install_dummy_profiler(monkeypatch):
    state = {"current_trace": None, "started": [], "stopped": 0, "servers": []}

    def start_trace(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state["current_trace"] = path
        state["started"].append(str(path))

    def stop_trace():
        path = state["current_trace"]
        if path is not None:
            (path / f"trace-{state['stopped']}.txt").write_text("trace", encoding="utf-8")
            state["current_trace"] = None
        state["stopped"] += 1

    def save_device_memory_profile(path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("memory-profile", encoding="utf-8")

    def start_server(port):
        state["servers"].append(port)

    monkeypatch.setattr("spikyeggroll.train.jax.profiler.start_trace", start_trace)
    monkeypatch.setattr("spikyeggroll.train.jax.profiler.stop_trace", stop_trace)
    monkeypatch.setattr(
        "spikyeggroll.train.jax.profiler.save_device_memory_profile",
        save_device_memory_profile,
    )
    monkeypatch.setattr("spikyeggroll.train.jax.profiler.start_server", start_server)
    return state


def test_train_startup_profile_writes_artifacts(monkeypatch, tmp_path):
    _install_dummy_dataset(monkeypatch)
    profiler_state = _install_dummy_profiler(monkeypatch)

    cfg = SNNConfig(
        dataset="mnist",
        model_name="mlp_snn",
        n_inputs=8,
        hidden_size=4,
        n_classes=2,
        timesteps=2,
        pop_size=4,
        rank=1,
        sigma=0.02,
        sigma_min=0.001,
        sigma_max=0.005,
        lr=0.001,
        batch_size=2,
        chunk_size=2,
        num_epochs=1,
        updates_per_epoch=2,
        sigma_warmup_epochs=0,
        log_interval=1,
        test_interval=1,
        checkpoint_interval=0,
        run_name="pytest-profile-startup",
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        profile_mode="startup",
        profile_server_port=9999,
        profile_max_snapshots=4,
        profile_eval_once=True,
    )

    train(cfg)

    startup_path = tmp_path / "logs" / "pytest-profile-startup.startup.jsonl"
    profile_summary = tmp_path / "logs" / "pytest-profile-startup.profile-summary.json"
    metrics_path = tmp_path / "logs" / "pytest-profile-startup.metrics.jsonl"
    profile_dir = tmp_path / "logs" / "profiles" / "pytest-profile-startup"
    trace_dir = tmp_path / "logs" / "traces" / "pytest-profile-startup" / "startup"

    assert startup_path.exists()
    startup_records = [json.loads(line) for line in startup_path.read_text().splitlines() if line.strip()]
    labels = [record["label"] for record in startup_records]
    assert "train() begin run_name=pytest-profile-startup" in labels
    assert "start metric written" in labels
    assert any("startup trace started" in label for label in labels)
    assert any("startup trace stopped" in label for label in labels)
    assert profile_summary.exists()
    assert any(profile_dir.glob("*.pb"))
    assert any(trace_dir.glob("*"))
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    epoch_record = next(record for record in records if record["event"] == "epoch")
    assert "timing_population_score_mean_s" in epoch_record
    assert "eval_test_total_s" in epoch_record
    assert any(record["event"] == "eval_profile" for record in records)
    assert profiler_state["servers"] == [9999]


def test_train_steady_state_profile_window_and_paths(monkeypatch, tmp_path):
    _install_dummy_dataset(monkeypatch)
    _install_dummy_profiler(monkeypatch)

    cfg = SNNConfig(
        dataset="mnist",
        model_name="mlp_snn",
        n_inputs=8,
        hidden_size=4,
        n_classes=2,
        timesteps=2,
        pop_size=4,
        rank=1,
        sigma=0.02,
        sigma_min=0.001,
        sigma_max=0.005,
        lr=0.001,
        batch_size=2,
        chunk_size=2,
        num_epochs=1,
        updates_per_epoch=3,
        sigma_warmup_epochs=0,
        log_interval=1,
        test_interval=0,
        checkpoint_interval=0,
        run_name="pytest-profile-steady",
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        profile_mode="steady_state",
        profile_warmup_updates=1,
        profile_updates_window=1,
        profile_max_snapshots=6,
    )

    train(cfg)

    startup_path = tmp_path / "logs" / "pytest-profile-steady.startup.jsonl"
    startup_records = [json.loads(line) for line in startup_path.read_text().splitlines() if line.strip()]
    labels = [record["label"] for record in startup_records]
    assert any("steady-state trace started" in label for label in labels)
    assert any("steady-state trace stopped" in label for label in labels)

    trace_dir = tmp_path / "logs" / "traces" / "pytest-profile-steady" / "steady_state"
    assert any(trace_dir.glob("*"))

    metrics_path = tmp_path / "logs" / "pytest-profile-steady.metrics.jsonl"
    epoch_record = next(
        record
        for record in (json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip())
        if record["event"] == "epoch"
    )
    assert "timing_total_update_mean_s" in epoch_record
    assert "timing_population_score_frac" in epoch_record
    assert (tmp_path / "logs" / "pytest-profile-steady.profile-summary.json").exists()
    assert (tmp_path / "ckpts").exists()


def test_train_selective_stage_perturbation_emits_metrics(monkeypatch, tmp_path):
    def loader():
        train_images = jnp.linspace(
            0.0, 1.0, 8 * 32 * 32 * 3, dtype=jnp.float32
        ).reshape(8, 32, 32, 3)
        train_labels = (jnp.arange(8) % 10).astype(jnp.int32)
        test_images = train_images[:4]
        test_labels = train_labels[:4]
        return train_images, train_labels, test_images, test_labels

    def encoder(images, timesteps, key):
        del key
        spikes = (images > 0.5).astype(jnp.float32)
        return jnp.broadcast_to(
            jnp.transpose(spikes, (0, 3, 1, 2))[:, None, :, :, :],
            (images.shape[0], timesteps, 3, 32, 32),
        )

    monkeypatch.setattr(
        "spikyeggroll.train.get_dataset_spec",
        lambda cfg: DatasetSpec(
            loader=loader,
            encoder=encoder,
            n_inputs=3072,
            in_channels=3,
            image_size=32,
        ),
    )

    cfg = SNNConfig(
        dataset="cifar10",
        model_name="spiking_resnet18",
        n_inputs=3072,
        in_channels=3,
        image_size=32,
        n_classes=10,
        timesteps=2,
        pop_size=4,
        rank=1,
        sigma=0.01,
        sigma_min=0.0025,
        sigma_max=0.012,
        lr=0.001,
        batch_size=2,
        chunk_size=2,
        num_epochs=2,
        updates_per_epoch=1,
        sigma_warmup_epochs=0,
        log_interval=1,
        test_interval=0,
        checkpoint_interval=0,
        run_name="pytest-selective-cifar",
        log_dir=str(tmp_path / "logs"),
        checkpoint_dir=str(tmp_path / "ckpts"),
        resnet_channels_base=8,
        selective_stage_perturbation=True,
        stage_perturbation_early_fraction=0.5,
        stage_perturbation_full_epoch_interval=8,
    )

    train(cfg)

    metrics_path = tmp_path / "logs" / "pytest-selective-cifar.metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    epoch_records = [r for r in records if r["event"] == "epoch"]

    assert len(epoch_records) == 2
    assert epoch_records[0]["selective_stage_perturbation"] is True
    assert epoch_records[0]["perturbation_phase"] == "early_selective"
    assert epoch_records[0]["cache_split"] == "after_stage2"
    assert epoch_records[0]["active_stage_groups"] == ["stage3", "head"]
    assert epoch_records[0]["active_param_fraction"] < 1.0
    assert epoch_records[1]["perturbation_phase"] == "mid_selective"
    assert epoch_records[1]["cache_split"] == "after_stage1"
    assert epoch_records[1]["active_stage_groups"] == ["stage2", "stage3", "head"]
