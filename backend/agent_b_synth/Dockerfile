FROM python:3.11-slim


COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY ./backend/agent_b_synth ./agent_b_synth

COPY ./backend/shared ./shared

# Agent B exposes the A2A server port
EXPOSE 8001

# Start the A2A server
CMD ["uvicorn", "agent_b_synth.main:app", "--host", "0.0.0.0", "--port", "8001"]