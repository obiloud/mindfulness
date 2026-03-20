FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY ./agent_b_synth ./agent_b_synth

# Agent B exposes the A2A server port
EXPOSE 8001

# Start the A2A server
# Note: Adjust the module path to where your A2AServer implementation lives
CMD ["python", "-m", "agent_b_synth.main"]