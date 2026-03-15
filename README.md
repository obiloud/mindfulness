# Mindfulness AI Agent

## Overview

This project provides a mindfulness coaching agent that:

- **Uses LangGraph** to orchestrate a reflection-capable agent.
- **Generates guided meditation scripts** via Hugging Face models.
- **Generates speech with Cartesia.ai** for natural-sounding audio sessions.
- **Exposes a FastAPI backend** for HTTP access.
- **Includes a React + Tailwind web UI**.
- **Provides a SwiftUI iOS client skeleton**.

## Backend (FastAPI + LangGraph)

- Entrypoint module: `api.py` (FastAPI app with:
  - `POST /v1/mindfulness/session` – conversational endpoint.
  - `POST /v1/mindfulness/audio` – Cartesia TTS streaming endpoint.
  - `GET /health` – health check.
- Agent orchestration:
  - `meditation_graph.py` – LangGraph-based reflection agent.
  - `meditation_agent.py` – LangChain tools and legacy agent wiring.
  - `story_generator_pipeline.py` – meditation script generator.
  - `cartesia_tts_client.py` – Cartesia.ai TTS wrapper.

### Environment and configuration

Create a `.env` file at the project root:

```env
HF_TOKEN=your_huggingface_token
CARTESIA_API_KEY=your_cartesia_api_key
API_HOST=0.0.0.0
API_PORT=8000
```

The `settings.py` module centralizes configuration and reads from `.env`.

### Run the API

Install Python dependencies (example, adjust to your environment):

```bash
pip install fastapi uvicorn "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres langchain langchain-huggingface python-dotenv pydantic pydantic_settings
```

Run the server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Web UI (React + Tailwind)

The web client lives in `web/` and is a Vite + React + Tailwind SPA.

### Setup

```bash
cd web
npm install
```

Optionally configure the backend base URL via `VITE_API_BASE_URL`:

```bash
echo 'VITE_API_BASE_URL="http://localhost:8000"' > .env.local
```

### Run the dev server

```bash
npm run dev
```

Then open the printed URL (default `http://localhost:5173`) in your browser.

## iOS Client (SwiftUI)

The iOS skeleton lives in `ios/`:

- `MindfulnessApp.swift` – app entry point.
- `ChatViewModel.swift` – networking and state.
- `ChatView.swift` – simple chat-style UI.

The view model reads the backend URL from the `API_BASE_URL` environment variable,
defaulting to `http://localhost:8000`.

Integrate these files into an Xcode SwiftUI project and ensure the app can
reach your FastAPI instance (e.g. via local network / simulator configuration).

## Architecture Summary

- **LLM + tools**:
  - Hugging Face endpoints (via `langchain-huggingface`) for meditation script generation.
  - A LangGraph graph (`meditation_graph.py`) that can clarify user intent, reflect on its answers, and generate a final transcript.
- **TTS**:
  - Cartesia.ai via `cartesia_tts_client.py` and the `/v1/mindfulness/audio` endpoint.
- **Clients**:
  - SPA web client in `web/`.
  - iOS SwiftUI client in `ios/`.

