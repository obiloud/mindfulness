# Mindfulness AI - Full-Stack Mindfulness Application

A modern, cross-platform mindfulness application built with a clean separation of concerns between backend and frontend services.

## Project Overview

Mindfulness AI is a full-stack application designed to provide users with guided meditation, breathing exercises, and mindfulness practices. The architecture leverages modern technologies to ensure scalability, maintainability, and performance across platforms.

## Project Structure

```
mindfulness-ai/
├── backend/            # FastAPI project
│   ├── agent_a_chat/   # AI chat agent for conversation-based meditation
│   │   ├── state.py    # Core state management for chat sessions
│   │   └── prompts/    # Prompt templates for conversation flows
│   │       ├── conversation.py # Conversation-specific prompt templates
│   │       └── __init__.py
│   │
│   ├── agent_b_synth/  # AI content synthesis agent for meditation content
│   │   ├── state.py    # Core state management for meditation sessions
│   │   └── prompts/    # Prompt templates for meditation content generation
│   │       ├── meditation.py # Meditation-specific prompt templates
│   │       ├── supervisor.py # Supervisor prompt templates
│   │       └── __init__.py
│   │
│   ├── node_evaluator.py # Node evaluation logic for processing meditation workflows
│   ├── prompts/        # Shared prompt templates
│   │   └── base.py     # Base prompt templates
│   └── workflow.py     # Workflow definitions for meditation sequences and transitions
│
├── client/             # Gleam/Lustre project (frontend)
│   ├── src/            # Gleam source files containing UI components and state logic
│   │   ├── api.gleam   # API interaction layer for frontend
│   │   ├── utils.gleam # Utility functions for common operations
│   │   ├── dom.gleam   # DOM manipulation and event handling
│   │   ├── theme.gleam # Theme and styling management
│   │   ├── cartesia.gleam # Cartesia integration for voice synthesis
│   │   ├── client.gleam # Main application entry point and state management
│   │
│   ├── ffi/            # JavaScript bridge files for audio processing and device interactions
│   │   └── audio_ffi.mjs # JavaScript bridge for audio processing and playback
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
- **Architecture**: Microservices pattern with two specialized AI agents
  - **Agent A (Chat Agent)**: Handles conversation-based meditation sessions using prompt templates for dialogue flows
  - **Agent B (Synthesis Agent)**: Generates meditation content including guided sessions and breathing exercises
- **Features**:
  - Real-time meditation session tracking
  - User authentication and session management
  - Audio streaming and playback
  - Data persistence and analytics
  - Workflow-based meditation sequences with AI prompt generation
  - Modular prompt system with shared base templates and agent-specific templates

### Core Orchestration via A2A Protocol

The application leverages the A2A (Agent-to-Agent) protocol as the core mechanism for orchestrating AI agent interactions. This protocol enables seamless coordination between specialized agents, allowing for a modular and scalable architecture where each agent performs a specific function.

Key aspects of A2A protocol usage:
- **Asynchronous Workflow Orchestration**: The chat agent (Agent A) initiates meditation sessions and triggers background content generation through the synthesis agent (Agent B) using the A2A protocol
- **Stateful Communication**: The A2A protocol maintains state across agent interactions, ensuring that generated content can be properly injected back into the conversation context
- **Callback-Based Integration**: When heavy-compute tasks (like meditation content generation) are completed, the A2A protocol uses callback mechanisms to notify the chat agent and update the session state
- **Modular Agent Design**: Each agent operates independently with its own state and logic, communicating through the A2A protocol to maintain a clean separation of concerns

This agent orchestration model enables the application to handle complex mindfulness workflows efficiently, with the ability to scale individual components independently while maintaining a cohesive user experience.

### Frontend (Gleam/Lustre)
- **Language**: Gleam (functional, type-safe language with strong compile-time guarantees)
- **Framework**: Lustre (Elm-like architecture with immutable state and reactive UI)
- **Features**:
  - Reactive UI components with automatic re-rendering
  - Type-safe state management
  - Seamless integration with audio processing via FFI
  - Cross-platform support (iOS via Capacitor)
  - Responsive design with static assets
  - Voice synthesis integration through Cartesia API
  - Modular component structure with clear separation of concerns

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
- Voice-based meditation guidance through Cartesia integration
- Asynchronous content generation with real-time state updates

## Future Roadmap

- Android platform support via Capacitor
- Voice-based meditation guidance
- AI-powered personalization of mindfulness content
- Community sharing and meditation challenges
- Wearable device integration (smartwatches)

## Contributing

Contributions are welcome! Please follow the standard contribution guidelines and ensure all changes adhere to the project's code style and architecture principles.

For more information on the technology stack and architecture, see the official documentation for FastAPI and Gleam.