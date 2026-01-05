import os
import netrc
import subprocess
import re

os.makedirs("logs", exist_ok=True)
os.makedirs("hpc", exist_ok=True)
assert os.environ.get("DATA_PATH") is not None, (
    "Please set the DATA_PATH environment variable"
)


def clean_environment():
    # 1. Scan code for actual imports
    print("🔍 Scanning code for imports...")

    # We add --ignore to skip virtual env folders that might contain legacy code
    subprocess.run(
        [
            "uvx",
            "pipreqs",
            ".",
            "--savepath",
            "clean-reqs.txt",
            "--force",
            "--ignore",
            ".venv",
        ],
        check=True,
    )

    print("🧹 Clearing old dependency list...")
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "r") as f:
            content = f.read()

        # Regex to remove the dependencies block
        new_content = re.sub(
            r"dependencies\s*=\s*\[.*?\]", "", content, flags=re.DOTALL
        )

        with open("pyproject.toml", "w") as f:
            f.write(new_content)
    else:
        print("⚠️ pyproject.toml not found. Creating a new one...")
        subprocess.run(["uv", "init", "."], check=True)

    print("📦 Installing clean dependencies...")

    if os.path.exists("clean-reqs.txt") and os.path.getsize("clean-reqs.txt") > 0:
        try:
            subprocess.run(["uv", "add", "-r", "clean-reqs.txt"], check=True)
        except subprocess.CalledProcessError:
            print("⚠️ Error adding packages. Check clean-reqs.txt for conflicts.")
    else:
        print("ℹ️  No imports detected in your code. Nothing to add.")

    if os.path.exists("clean-reqs.txt"):
        os.remove("clean-reqs.txt")

    print("✅ Environment sync complete! Unused packages removed.")


def wandb_logged_in():
    netrc_path = os.path.expanduser("~/.netrc")
    if not os.path.exists(netrc_path):
        return False

    auth = netrc.netrc(netrc_path).authenticators("api.wandb.ai")
    return bool(auth)


assert wandb_logged_in(), "Please log in to wandb using `wandb login`"
clean_environment()

print("Setup complete.")
