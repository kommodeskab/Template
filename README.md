# PyTorch Lightning Hydra Template

## Creating a New Project from This Template

Use [copier](https://copier.readthedocs.io) to generate a new project:

```bash
uvx copier copy <template-repo-url> <destination>
```

You will be prompted for:

- `project_name` — Python package name in `snake_case` (e.g. `my_project`)

After generation, install dependencies:

```bash
cd <destination>
uv sync
```

### (Optional) Install pre-commit

```bash
pre-commit install
```

## Running Experiments

- Run an experiment:

```bash
python main.py experiment=<experiment-name>
```

- Override arguments on the command line:

```bash
python main.py experiment=<experiment-name> data.batch_size=2
```

- Run with compiled model:

```bash
python main.py experiment=<experiment-name> compile=True
```

- Continue a previous training run using W&B run ID + checkpoint file:

```bash
python main.py experiment=<experiment-name> continue_from_id=<id> ckpt_filepath=<filepath>
```

- Start a new run but initialize from another checkpoint:

```bash
python main.py experiment=<experiment-name> ckpt_filepath=<filepath>
```

## Invoke Tasks

Run tasks with:

```bash
invoke <task-name>
```

Available tasks are defined in `tasks.py`. For example:

```bash
invoke format
```

## Fill out `.env`

Add relevant API keys and paths (local data path, Weights & Biases, Hugging Face, etc.).
Typical keys: `DATA_PATH`, `WANDB_ENTITY`, `WANDB_API_KEY`, `HF_TOKEN`.

## Potential Improvements

- Authenticate external services if needed (for example, `wandb login`).
- Add a troubleshooting section for missing env vars, path issues, or CUDA/GPU setup.
