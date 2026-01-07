from invoke import task, Context


@task
def stopcontainers(c: Context):
    """Stop all Docker containers."""
    c.run("echo Stopping following containers:")
    c.run("docker stop $(docker ps -q)")
    
@task
def cleandocker(c: Context, all: bool = False):
    """Remove (unused) Docker containers, images, and volumes. Pass -all to remove everything."""
    c.run(f"docker system prune {'-a' if all else ''}")

@task
def image(c: Context):
    """Build the Docker development container image."""
    c.run("docker build -f .devcontainer/Dockerfile -t main-image .")
    
@task
def dockermain(c: Context, extra: str = ""):
    """Run main.py inside the Docker development container. Specify the 'extra' argument to add extra command line arguments."""
    c.run(
        "docker run --rm "
        "-v $(pwd):/app "
        "-v uv-venv:/app/.venv "
        "-v uv-cache:/root/.cache/uv "
        f"main-image {extra}"
    )

@task
def format(c: Context):
    """Format code using ruff."""
    c.run("uv run ruff check . --fix")
    

@task
def typing(c: Context, filename: str | None = None):
    """Check typing using mypy."""
    filename = filename.strip() if filename else "."
    
    c.run(f"uv run mypy {filename}")


@task
def python(ctx: Context):
    """ """
    ctx.run("which python")
    ctx.run("python --version")


@task
def build(c: Context):
    """Build (sync) the environment from pyproject.toml."""
    c.run("echo Syncing the environment...")
    c.run("uv sync")

    # also make required directories
    c.run("mkdir -p logs hpc data")

    # make .env file
    c.run("echo Creating .env file...")
    with open(".env", "w") as f:
        f.write("DATA_PATH=...\n")
        f.write("WANDB_API_KEY=...\n")
        f.write("ZOTERO_API_KEY=...\n")
    c.run("echo .env file created with WANDB_API_KEY and ZOTERO_API_KEY variables.")


@task
def update(c: Context):
    """
    Auto-detect imports and update pyproject.toml.
    WARNING: This may overwrite manual version constraints.
    """
    c.run("echo Detecting dependencies from source code...")
    c.run("uvx pipreqs --force --ignore .venv")
    c.run("uv add -r requirements.txt")
    c.run("rm requirements.txt")


@task
def submit(
    c: Context,
    experiment="",
    gpu="gpuv100",
    ngpus=1,
    ncores=4,
    mem=4,
    walltime="3:00",
    jobname=None,
):
    """
    Submit a training job to HPC using bsub.

    Args:
        experiment: Name of the experiment config (without .yaml)
        gpu: GPU type (gpuv100 or gpua100)
        ngpus: Number of GPUs to request
        ncores: Number of CPU cores
        mem: Memory per core in GB
        walltime: Wall time in HH:MM format
        jobname: Custom job name (defaults to experiment name)

    Example:
        invoke submit --experiment=template
        invoke submit --experiment=myexp --gpu=gpua100 --walltime=24:00
    """
    import tempfile
    import os

    # Use experiment name as2 job name if not provided
    if jobname is None:
        jobname = experiment

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
    #BSUB -o hpc/output_%J.out 
    #BSUB -e hpc/error_%J.err 

    module load python3/3.12.4
    source .venv/bin/activate
    python main.py {experiment}
    """

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script_content)
        temp_script = f.name

    try:
        # Submit the job
        c.run(f"bsub < {temp_script}")
        print(f"\n✓ Job '{jobname}' submitted with experiment={experiment}")
        print(f"  GPU: {gpu}, Cores: {ncores}, Memory: {mem}G, Walltime: {walltime}")
    finally:
        # Clean up temporary file
        os.unlink(temp_script)


@task
def status(c: Context, user=None):
    """Check status of submitted jobs."""
    if user:
        c.run(f"bjobs -u {user}")
    else:
        c.run("bjobs")


@task
def logs(c: Context, jobid=None, tail=50):
    """
    View logs from HPC jobs.

    Args:
        jobid: Job ID to view logs for (if None, shows latest)
        tail: Number of lines to show (default: 50)
    """
    if jobid:
        c.run(f"tail -n {tail} hpc/output_{jobid}.out")
        print("\n--- Errors ---")
        c.run(f"tail -n {tail} hpc/error_{jobid}.err", warn=True)
    else:
        # Show most recent log files
        print("Most recent output:")
        c.run(f"ls -t hpc/output_*.out | head -1 | xargs tail -n {tail}", warn=True)
        print("\nMost recent errors:")
        c.run(f"ls -t hpc/error_*.err | head -1 | xargs tail -n {tail}", warn=True)
