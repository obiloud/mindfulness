FROM python:3.11-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY ./backend/agent_a_chat ./agent_a_chat

COPY ./backend/shared ./shared

EXPOSE 8000

# Start the FastAPI gateway
CMD ["uvicorn", "agent_a_chat.main:app", "--host", "0.0.0.0", "--port", "8000"]