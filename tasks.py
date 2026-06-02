from invoke import task, Context
from typing import Optional
import os
from pathlib import Path
from module_name.utils import get_data_path, get_logs_path


@task
def format(c: Context):
    """Format code using ruff."""
    c.run("uv run ruff check . --fix")


@task
def typing(c: Context, filename: Optional[str] = None):
    """Check typing using mypy."""
    filename = filename.strip() if filename else "."

    c.run(f"uv run mypy {filename}")


@task
def test(c: Context):
    """Run tests using pytest."""
    c.run("uv run pytest src/module_name/tests")


@task
def python(ctx: Context):
    """ """
    ctx.run("which python")
    ctx.run("python --version")


@task
def build(c: Context) -> None:
    """Build (sync) the environment and initialize .env placeholders."""
    c.run("echo 'Syncing the environment...'")
    c.run("uv sync")
    c.run("pre-commit install")

    env_path = Path(".env")
    items = {
        "DATA_PATH": "...",
        "WANDB_ENTITY": "...",
        "WANDB_API_KEY": "...",
        "ZOTERO_API_KEY": "...",
        "ZOTERO_USER_ID": "...",
        "PAPER_PATH": "...",
        "HF_TOKEN": "...",
    }

    curr_vars = set()
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                key, _, _ = line.partition("=")
                curr_vars.add(key.strip())

    missing_items = {k: v for k, v in items.items() if k not in curr_vars}

    if missing_items:
        with env_path.open("a", encoding="utf-8") as f:
            for key, val in missing_items.items():
                f.write(f"{key}={val}\n")

    # get the DATA_PATH and make the folder if it doesn't exist
    data_path = get_data_path()
    os.makedirs(data_path, exist_ok=True)

    logs_path = get_logs_path()
    os.makedirs(logs_path / "wandb", exist_ok=True)
    os.makedirs(logs_path / "hpc", exist_ok=True)


@task
def submit(
    c: Context,
    command: str,
    jobname: str,
    gpu="gpuv100",
    ngpus=1,
    ncores=4,
    mem=4,
    walltime="3:00",
):
    # make sure "logs/hpc" exists

    """
    Submit a training job to HPC using bsub.

    Args:
        command: The command to run in the job
        gpu: GPU type (gpuv100 or gpua100)
        ngpus: Number of GPUs to request
        ncores: Number of CPU cores
        mem: Memory per core in GB
        walltime: Wall time in HH:MM format
        jobname: Custom job name (defaults to experiment name)

    Example:
        >>> invoke submit --command="python main.py +experiment=dummy +trainer.max_steps=100" --gpu=gpua100 --walltime=24:00
    """
    import tempfile
    import os

    # Create a temporary bash script with the specified parameters
    script_content = f"""#!/bin/sh

    # SET JOB NAME
    #BSUB -J {jobname}

    # select gpu, choose gpuv100 or gpua100 (best)
    #BSUB -q {gpu}

    # number of GPUs to use
    #BSUB -gpu "num={ngpus}:mode=exclusive_process"

    # number of cores to use
    #BSUB -n {ncores}

    # gb memory per core
    #BSUB -R "rusage[mem={mem}G]"
    # cores is on the same slot
    #BSUB -R "span[hosts=1]"

    # walltime
    #BSUB -W {walltime}
    #BSUB -o logs/hpc/output_%J.out
    #BSUB -e logs/hpc/error_%J.err

    source .venv/bin/activate
    {command}
    """

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        # Submit the job
        c.run(f"bsub < {temp_script}")
        print(f"\n✓ Job '{jobname}' submitted with command:\n  {command}")
        print(f"  GPU: {gpu}, Cores: {ncores}, Memory: {mem}G, Walltime: {walltime}")
    finally:
        # Clean up temporary file
        os.unlink(temp_script)


@task
def submit_experiment(
    c: Context,
    experiment: str,
    jobname: str,
    gpu="gpuv100",
    ngpus=1,
    ncores=4,
    mem=4,
    walltime="3:00",
):
    command = f"uv run python main.py {experiment}"
    submit(c, command, jobname, gpu, ngpus, ncores, mem, walltime)


@task
def status(c: Context, user=None):
    """Check status of submitted jobs."""
    if user:
        c.run(f"bjobs -u {user}")
    else:
        c.run("bjobs")


@task
def buildsweep(c: Context, name: str):
    """
    Initialize a Weights & Biases sweep from a YAML configuration file.

    Args:
        name (str): Name of the sweep configuration file (without .yaml extension)
    """
    # initialize the sweep
    c.run(f"WANDB_DIR=logs uv run wandb sweep configs/sweeps/{name}.yaml")


@task
def runsweep(c: Context, name: str):
    """
    Run a Weights & Biases sweep agent for the specified sweep ID.

    Args:
        name (str): The name of the sweep to run the agent for
    """
    c.run(f"WANDB_DIR=logs uv run wandb agent {name}")


@task
def submitsweep(
    c: Context,
    name: str,
    jobname: str,
    gpu="gpuv100",
    ngpus=1,
    ncores=4,
    mem=4,
    walltime="3:00",
):
    """Submit a Weights & Biases sweep agent as an HPC job.

    Args:
        c (Context): _invoke_ context
        name (str): The name of the sweep to run the agent for
        jobname (str): The name of the HPC job
        gpu (str, optional): GPU type (gpuv100 or gpua100). Defaults to "gpuv100".
        ngpus (int, optional): Number of GPUs to request. Defaults to 1.
        ncores (int, optional): Number of CPU cores. Defaults to 4.
        mem (int, optional): Memory per core in GB. Defaults to 4.
        walltime (str, optional): Wall time in HH:MM format. Defaults to "3:00".
    """
    command = f"WANDB_DIR=logs uv run wandb agent {name}"
    submit(c, command, jobname, gpu, ngpus, ncores, mem, walltime)


@task
def logs(c: Context, jobid=None, tail=50):
    """
    View logs from HPC jobs.

    Args:
        jobid: Job ID to view logs for (if None, shows latest)
        tail: Number of lines to show (default: 50)
    """
    if jobid:
        c.run(f"tail -n {tail} logs/hpc/output_{jobid}.out")
        print("\n--- Errors ---")
        c.run(f"tail -n {tail} logs/hpc/error_{jobid}.err", warn=True)
    else:
        # Show most recent log files
        print("Most recent output:")
        c.run(
            f"ls -t logs/hpc/output_*.out | head -1 | xargs tail -n {tail}", warn=True
        )
        print("\nMost recent errors:")
        c.run(f"ls -t logs/hpc/error_*.err | head -1 | xargs tail -n {tail}", warn=True)


@task
def coverage(c: Context):
    """Generate code coverage report."""
    c.run("coverage run --source=src -m pytest")
    c.run("coverage report -m")
