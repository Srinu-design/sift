FROM python:3.12.15-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
# --no-cache: uv's download cache (~1.2 GB) would otherwise stay in this layer's build cache on
# the 19 GB host that builds the image; a uv.lock change is rare enough to download again.
RUN uv sync --frozen --no-dev --extra models --no-install-project --no-cache

FROM python:3.12.15-slim
# The Hugging Face cache is a volume; created here, owned by app, so a fresh volume is writable.
RUN useradd --create-home app && install -d -o app -g app /home/app/.cache/huggingface
WORKDIR /app
# Dependencies only (~2 GB with CPU torch): this layer changes only when uv.lock does, so a
# release that changes Sift's code adds megabytes to the host's disk, not gigabytes.
COPY --from=build /app/.venv /app/.venv
COPY migrations ./migrations
COPY alembic.ini ./
COPY src ./src
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app/src PYTHONUNBUFFERED=1
USER app
CMD ["uvicorn", "sift.models_svc.app:app", "--host", "0.0.0.0", "--port", "8000"]
