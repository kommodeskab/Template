# Lightweight image for running training
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Ensure Python output is sent straight to terminal without buffering
ENV PYTHONUNBUFFERED=1

# Install git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the location for the virtual environment explicitly
# We put it in /opt/venv so it isn't hidden when we mount our code to /app
ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
# Add the venv to the PATH so 'python' and 'uv' work automatically
ENV PATH="/opt/venv/bin:$PATH"
# Use copy mode for Docker
ENV UV_LINK_MODE=copy

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (but not the project root yet)
# --frozen: ensures lockfile is respected strictly
# --no-install-project: only installs dependencies, keeping this layer cached if code changes
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project

# Copy the rest of the application code
COPY . .

# Now install the project itself
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen

# Set the default command
ENTRYPOINT ["uv", "run", "main.py"]