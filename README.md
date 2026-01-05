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

### 3. Workflow: add dependency
To add a new dependency to your project, use the following command:
```bash
uv add <package-name>
```

## WandB
Log into WandB using:
```
wandb login
```

## Check formatting
To check code formatting, use:
