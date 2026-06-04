import os
import random
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf

from module_name.datasets import DummyDataset
from module_name.utils import (
    temporary_seed,
    get_context,
    get_current_time,
    get_environment_variable,
    get_data_path,
    get_logs_path,
    instantiate_callbacks,
    get_batch_from_dataset,
)


def test_temporary_seed():
    # Store initial states
    initial_random = random.getstate()
    initial_torch = torch.random.get_rng_state()

    with temporary_seed(42):
        # State inside temporary_seed
        inside_random_val = random.random()
        inside_numpy_val = np.random.rand()
        inside_torch_val = torch.randn(1).item()

    # Outside temporary_seed, values should be different from seeded values.
    # But let's verify setting the same seed again yields identical values.
    with temporary_seed(42):
        assert random.random() == inside_random_val
        assert np.random.rand() == inside_numpy_val
        assert torch.randn(1).item() == inside_torch_val

    # Verify states are restored
    assert random.getstate() == initial_random
    assert torch.equal(torch.random.get_rng_state(), initial_torch)


def test_get_context():
    # With deterministic=True
    ctx = get_context(seed=42, deterministic=True)
    with ctx:
        val1 = random.random()
    with get_context(seed=42, deterministic=True):
        val2 = random.random()
    assert val1 == val2

    # With deterministic=False (should behave as nullcontext, not setting seed)
    random.seed(123)
    with get_context(seed=42, deterministic=False):
        val3 = random.random()
    assert val3 != val1


def test_get_current_time():
    time_str = get_current_time()
    assert isinstance(time_str, str)
    assert len(time_str) == 12
    assert time_str.isdigit()


def test_get_environment_variable():
    name = "TEST_TEMP_VAR"
    assert get_environment_variable(name, "default_val") == "default_val"
    os.environ[name] = "custom_val"
    try:
        assert get_environment_variable(name, "default_val") == "custom_val"
    finally:
        del os.environ[name]


def test_get_data_path():
    os.environ["DATA_PATH"] = "my_data_dir"
    try:
        assert get_data_path() == Path("my_data_dir")
    finally:
        del os.environ["DATA_PATH"]


def test_get_logs_path():
    assert get_logs_path() == Path("logs")


def test_instantiate_callbacks():
    # Pass None
    assert instantiate_callbacks(None) == []

    # Config dict for instantiating a dummy callback
    config = OmegaConf.create(
        {
            "stop_training": {
                "_target_": "module_name.callbacks.StopTrainingCallback",
                "key": "stop_reason",
                "value": 1.0,
            }
        }
    )
    callbacks = instantiate_callbacks(config)
    assert len(callbacks) == 1
    from module_name.callbacks import StopTrainingCallback

    assert isinstance(callbacks[0], StopTrainingCallback)


def test_get_batch_from_dataset():
    dataset = DummyDataset(size=5)
    batch = get_batch_from_dataset(dataset, batch_size=2)
    assert "input" in batch
    assert "target" in batch
    assert batch["input"].shape[0] == 2
