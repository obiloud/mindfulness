# Mindfulness AI - Full-Stack Mindfulness Application

A modern, cross-platform mindfulness application built with a clean separation of concerns between backend and frontend services.

## Project Overview

Mindfulness AI is a full-stack application designed to provide users with guided meditation, breathing exercises, and mindfulness practices. The architecture leverages modern technologies to ensure scalability, maintainability, and performance across platforms.

## Project Structure

```
mindfulness-ai/
├── backend/            # FastAPI project
│   ├── app/            # Python logic for API endpoints, data processing, and business rules
│   ├── state.py        # Core state management for meditation sessions and user data
│   ├── workflow.py     # Workflow definitions for meditation sequences and transitions
│   ├── prompts/        # Prompt templates for AI-generated meditation content
│   │   ├── base.py     # Base prompt templates
│   │   └── meditation.py # Meditation-specific prompt templates
│   ├── node_evaluator.py # Node evaluation logic for processing meditation workflows
│   ├── workflow_mermaid.png # Visual representation of workflow diagrams
│   └── pytest.ini      # Configuration for Pytest testing framework
│
├── client/             # Gleam/Lustre project (frontend)
│   ├── src/            # Gleam source files containing UI components and state logic
│   │   ├── api.gleam   # API interaction layer for frontend
│   │   ├── audio_ffi.mjs # JavaScript bridge for audio processing and playback
│   │   ├── client.gleam # Main application entry point and state management
│   │
│   ├── ffi/            # JavaScript bridge files for audio processing and device interactions
│   │   └── audio_bridge.js (mocks)
│   │
│   ├── ios/            # Capacitor iOS platform folder for native iOS integration
│   │
│   ├── gleam.toml      # Configuration for Gleam compiler and build settings
│   ├── manifest.toml   # Application manifest and metadata
│   ├── static/         # Static assets (CSS, JS, HTML)
│   │   ├── index.css
│   │   ├── index.js
│   │   └── index.html
│   │
│   └── build/          # Build artifacts (compiled output)
│       ├── gleam-prod-erlang.lock
│       ├── gleam-dev-javascript.lock
│       ├── gleam-dev-erlang.lock
│       ├── gleam-prod-javascript.lock
│       └── gleam-lsp-javascript.lock
│
└── docker-compose.yml # Defines the service orchestration for backend and frontend
````

## Technology Stack

### Backend (Python)
- **Framework**: FastAPI (for RESTful APIs with async support)
- **Features**: 
  - Real-time meditation session tracking
  - User authentication and session management
  - Audio streaming and playback
  - Data persistence and analytics
  - Workflow-based meditation sequences with AI prompt generation

### Frontend (Gleam/Lustre)
- **Language**: Gleam (functional, type-safe language with strong compile-time guarantees)
- **Framework**: Lustre (Elm-like architecture with immutable state and reactive UI)
- **Features**:
  - Reactive UI components with automatic re-rendering
  - Type-safe state management
  - Seamless integration with audio processing via FFI
  - Cross-platform support (iOS via Capacitor)
  - Responsive design with static assets

### Cross-Platform Integration
- **Audio Bridge**: JavaScript FFI layer enables the frontend to interact with audio processing libraries
- **Platform Support**: iOS via Capacitor, with future expansion to Android
- **State Management**: Centralized, immutable state model following The Elm Architecture (TEA)

## Development Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- Gleam compiler (v0.20+)
- Docker (for local development and testing)

### Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/mindfulness-ai.git
   cd mindfulness-ai
   ```

2. **Set up backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Set up frontend**
   ```bash
   cd ../client
   gleam build
   ```

4. **Start services**
   ```bash
   docker-compose up --build
   ```

## Key Features

- Guided meditation sessions with customizable duration and focus
- Breathing exercises with real-time feedback
- Progress tracking and analytics
- Offline mode with local storage
- Device-specific audio playback via FFI integration
- AI-generated meditation content based on user preferences

## Future Roadmap

- Android platform support via Capacitor
- Voice-based meditation guidance
- AI-powered personalization of mindfulness content
- Community sharing and meditation challenges
- Wearable device integration (smartwatches)

## Contributing

Contributions are welcome! Please follow the standard contribution guidelines and ensure all changes adhere to the project's code style and architecture principles.

For more information on the technology stack and architecture, see the official documentation for FastAPI and Gleam.