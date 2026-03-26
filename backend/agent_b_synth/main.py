import uvicorn
from fastapi import FastAPI
from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

# Import your agent executor and agent card from your LangGraph definition
from .agent_executor import PulseSynthExecutor
from .agent_card import PULSE_SYNTH_CARD

# Set up the required A2A server components
task_store = InMemoryTaskStore()

request_handler = DefaultRequestHandler(
    agent_executor=PulseSynthExecutor(),
    task_store=task_store
)

# Initialize the A2A FastAPI Wrapper
a2a_app = A2AFastAPIApplication(
    agent_card=PULSE_SYNTH_CARD,
    http_handler=request_handler
)

# Attach the A2A routes to a standard FastAPI app
app = FastAPI()
a2a_app.add_routes_to_app(app)

if __name__ == "__main__":
    # Ensure this matches the port exposed in your docker-compose or Dockerfile
    uvicorn.run(app, host="0.0.0.0", port=8000)
