FROM python:3.11-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files first for caching
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Create a directory for the model cache
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

# Set the environment variable so FastEmbed knows where to look
ENV FASTEMBED_CACHE_PATH=/app/model_cache

# (Optional) Pre-download the model during build to save time on first start
# This makes the image larger but the first request faster
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY ./backend/agent_a_chat ./agent_a_chat

COPY ./backend/shared ./shared

EXPOSE 8000

# Start the FastAPI gateway
CMD ["uvicorn", "agent_a_chat.main:app", "--host", "0.0.0.0", "--port", "8000"]