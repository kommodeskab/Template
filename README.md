# Template
 Template for deep learning projects using Pytorch Lightning and Hydra on DTU HPC. 

## Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```

## Initialize Environment
To build the environment using `uv` and invoke, run:
```bash
uvx invoke build
```

## Update the environment
Have you started using a new package and want to update the environment? Run:
```bash
uvx invoke update
```
or 
```bash
uvx invoke update build
```
To also rebuild the environment.

## Activate Environment
To activate the environment, run:
```bash
source .venv/bin/activate
```


## WandB
Log into WandB using:
```
wandb login
```

## Data path
Set an environment variable `DATA_PATH` to point to the data directory:
```
export DATA_PATH="/path/to/your/data"
```

## Check styling using Ruff
To check the code styling (before committing) using Ruff, run:
```bash
uv run ruff check .
```
or to automatically fix issues, run:
```bash
uv run ruff format .
```

## Check typing using MyPy
To check the code typing using MyPy, run:
```bash
uv run mypy <path-to-file>
```