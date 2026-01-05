from invoke import task, Context


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
    c.run(
        "uvx pipreqs . --savepath requirements.txt --force --ignore .venv --mode=no-pin"
    )
    c.run("uv add -r requirements.txt")
    c.run("rm requirements.txt")
