import shutil
from pathlib import Path
from hydra import compose, initialize_config_dir
from main import my_app


def test_main():
    # Get absolute path to the configs folder using pathlib
    config_dir = Path(__file__).resolve().parents[3] / "configs"

    # Track checkpoints existence before test to avoid deleting user files
    checkpoints_dir = Path(__file__).resolve().parents[3] / "checkpoints"
    had_checkpoints = checkpoints_dir.exists()

    try:
        # Initialize Hydra from the config directory and compose the config directly
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(
                config_name="config",
                overrides=[
                    "trainer=cpu",
                    "data=dummy_data",
                    "callbacks=standard",
                    "model=dummy_model",
                    "logger=default",
                    "project_name=test_experiment",
                    "task_name=test_task",
                    "compile=False",
                    "phase=train",
                    "logger.offline=True",
                    "trainer.fast_dev_run=True",
                ],
            )

            # Call the undecorated my_app function directly
            my_app.__wrapped__(cfg)
    finally:
        # Clean up the checkpoints directory if it was created during the test
        if not had_checkpoints and checkpoints_dir.exists():
            shutil.rmtree(checkpoints_dir)
