FROM python:3.12.15-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra models --no-install-project
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --frozen --no-dev --extra models --no-editable

FROM python:3.12.15-slim
# The Hugging Face cache is a volume; created here, owned by app, so a fresh volume is writable.
RUN useradd --create-home app && install -d -o app -g app /home/app/.cache/huggingface
WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/migrations /app/migrations
COPY --from=build /app/alembic.ini /app/alembic.ini
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER app
CMD ["uvicorn", "sift.models_svc.app:app", "--host", "0.0.0.0", "--port", "8000"]
