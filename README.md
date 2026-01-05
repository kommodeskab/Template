# Template
 Template for deep learning projects using Pytorch Lightning and Hydra on DTU HPC. 

## 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Initialize Environment
```bash
uvx invoke build
```
This runs the function `build.py` as seen in `tasks.py`. It is probably a good idea to inspect the functions in `tasks.py` before running them.


## Add dependency (package)
```bash
uv add <package-name>
```

## WandB
Log into WandB using:
```bash
wandb login
```

## Check formatting
To check (and automatically fix some) code formatting, use:
```bash
uvx invoke format
```

## Check typing
To check typing using mypy, use:
```bash
uvx invoke typing --filename <path-to-file-or-directory>
```
You can omit the `--filename` argument to check the entire codebase.

## Run an experiment
To run an experiment, use:
```bash
python src/main.py --config-name=<config-file-name>
```
You can add additional overrides as needed using Hydra. For example, to change the batch size, use:
```bash
python src/main.py --config-name=<config-file-name> data.batch_size=64
```

