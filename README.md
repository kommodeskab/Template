# Template
 Template for deep learning projects using Pytorch Lightning and Hydra on DTU HPC. 

## Clone repo
```bash
git clone https://github.com/kommodeskab/Template.git
```
Remember that you have to login. This can be done very easily with the Github CLI. Look it up. 

## New branch
It is FORBIDDEN (literally) to push to main. Therefore, when you want to make a new change, make a new branch, for example like this:
```bash
git branch -b <new-branch-name>
```
You can make all the changes you want on this branch, then commit and push.

## Make changes to main
As mentioned, you cannot push directly to main. Instead, push your changes to your own branch. If you want to merge these changes into main, then (after having pushed) go to the Github page for the Repo and find your branch. There should be a yellow/green banner saying "`<new-branch-name> had recent pushes`" with a button "Compare & pull request". Click that button, write a description of your changes, and create a pull request. After that, someone (maybe you) can review the changes and merge them into main. You can now see the changes on main:
```bash
git checkout main
git pull origin main
```
Optionally, if your own branch was made to make a new specific change, you can delete your branch both locally and on Github:
```bash
git branch -d <new-branch-name>     # delete locally
git push origin --delete <new-branch-name> # delete on Github
```
Otherwise, you can choose to "reset" or update your branch as described below.

## "Reset" you branch
Have your branced diverged from main? i.e. there have been some changes on main that you would like to also have on your branch? Then type:
```bash
git checkout <new-branch-name>
git merge main # or 'git rebase main'
```
Alternatively, if you want to throw out all your changes and forget them (note: all your changes on the branch will be lost):
```bash
git checkout <new-branch-name>
git reset --hard origin/main
```

## Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Initialize Environment
To build the environment, run:
```bash
uvx invoke build
```
This runs the function `build.py` as seen in `tasks.py`. It is probably a good idea to inspect the functions in `tasks.py` before running them.
Also remember to activate the environment after building it:
```bash
source .venv/bin/activate
```
Almost all commands assume that the environment is activated.

## Add dependency (package)
If you want to add a new dependency (package) to the environment, use:
```bash
uv add <package-name>
```

## WandB
Log into WandB by pasting the API key into the .env file.
The API key can be found at: [wandb.ai/authorize](https://wandb.ai/authorize)

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
python main.py --config-name=<config-file-name>
```
You can add additional overrides as needed using Hydra. For example, to change the batch size, use:
```bash
python main.py --config-name=<config-file-name> data.batch_size=64
```

## Run the dummy example
You can try out the dummy example to get started by running:
```bash
python main.py --config-name=dummy
```

## Pytest and coverage
To run the tests using pytest, use:
```bash
pytest
```
Test a specific file:
```bash
pytest <path-to-file>
```

Run a specific test function in a file:
```bash
pytest <path-to-file>::<test-function-name>
```

To run with print statements enabled, use:
```bash
pytest -s
```

To see how much of the code is covered by tests, use:
```bash
coverage run -m pytest
coverage report -m
``` 