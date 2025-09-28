import os
import netrc

os.makedirs("logs", exist_ok=True)
os.makedirs("hpc", exist_ok=True)
assert os.environ.get("DATA_PATH") is not None, "Please set the DATA_PATH environment variable"

def wandb_logged_in():
    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        return False

    auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
    return bool(auth)

assert wandb_logged_in(), "Please log in to wandb using `wandb login`"

print("Setup complete.")