FROM ghcr.io/astral-sh/uv:latest AS uv_builder

FROM python:3.11-slim

COPY --from=uv_builder /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

# Create a directory for the model cache
RUN mkdir -p /app/model_cache && chmod 777 /app/model_cache

# Set the environment variable so FastEmbed knows where to look
ENV FASTEMBED_CACHE_PATH=/app/model_cache

# (Optional) Pre-download the model during build to save time on first start
# This makes the image larger but the first request faster
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

COPY ./backend/agent_b_synth ./agent_b_synth

COPY ./backend/shared ./shared

# (Optional) Include the sqlalchemy entities
# COPY ./backend/entities ./entities

# Agent B exposes the A2A server port
EXPOSE 8001

# Start the A2A server
CMD ["uvicorn", "agent_b_synth.main:app", "--host", "0.0.0.0", "--port", "8001"]