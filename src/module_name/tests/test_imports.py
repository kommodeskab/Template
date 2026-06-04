def test_imports():
    # Import types from core module package
    from module_name import (
        Sample,  # noqa: F401
        Batch,  # noqa: F401
        ModelOutput,  # noqa: F401
        LossOutput,  # noqa: F401
        StepOutput,  # noqa: F401
    )

    # Import data modules & datasets
    from module_name.data_modules import BaseDM  # noqa: F401
    from module_name.datasets import BaseDataset, DummyDataset  # noqa: F401

    # Import lightning modules & networks
    from module_name.lightning_modules import BaseLightningModule, DummyModule  # noqa: F401
    from module_name.networks import DummyNetwork, PretrainedModel  # noqa: F401

    # Import losses
    from module_name.losses import (
        BaseLossFunction,  # noqa: F401
        WeightedLoss,  # noqa: F401
        MSELoss,  # noqa: F401
        SmoothL1Loss,  # noqa: F401
        L1Loss,  # noqa: F401
        SNRLoss,  # noqa: F401
    )

    # Import callbacks
    from module_name.callbacks import (
        EMACallback,  # noqa: F401
        LogLossCallback,  # noqa: F401
        LogGradsCallback,  # noqa: F401
        BatchesPerSecondCallback,  # noqa: F401
        ParameterCountCallback,  # noqa: F401
        WandbWatchCallback,  # noqa: F401
        LogGraphCallback,  # noqa: F401
        StopTrainingCallback,  # noqa: F401
        MetricsCallback,  # noqa: F401
    )
