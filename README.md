# Template
 Template for deep learning projects using Pytorch Lightning and Hydra on DTU HPC. 

## Setting up environment
### 1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```
### 2. Initialize Environment

To set up the virtual environment and install dependencies defined in `uv.lock`:

```bash
uv sync

```
*This command automatically creates a virtual environment in `.venv` and syncs it with the lockfile.*

---

### 3. Workflow: Sync Dependencies to Source Code

If you have written new code or want to reset `pyproject.toml` to match **only** the libraries currently imported in your Python files, follow these steps.

**Step A: Scan imports**
Use `pipreqs` (via `uvx`, so no installation is required) to generate a temporary requirements file based on your actual imports:

```bash
uvx pipreqs . --savepath clean-reqs.txt --force --ignore ".venv"
```

**Step B: Update Project Dependencies**
Add these detected packages to your project.

> **Note:** If you want to strictly remove unused packages, delete the `dependencies = [...]` section in `pyproject.toml` before running the command below.

```bash
uv add -r clean-reqs.txt

```

**Step C: Cleanup**
Remove the temporary file:

```bash
rm clean-reqs.txt

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

## Setup
Run the `setup.py` to setup the project (setting up the environment comes later).
```
python setup.py
```
