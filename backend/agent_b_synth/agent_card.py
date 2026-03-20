from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill
)

PULSE_SYNTH_CARD = AgentCard(
    # 'name' is the unique system identifier; 'display_name' is for humans
    name="pulse-lotus-synth-v1",
    display_name="Pulse Lotus: Meditation Synthesizer",
    version="1.0.0",

    # The reachable URL for this agent instance
    url="http://pulse-synth-agent:8001",

    description="A high-fidelity reflection agent for multi-pass mindfulness transcript generation.",

    # 'capabilities' must be an AgentCapabilities instance or dict
    capabilities=AgentCapabilities(
        streaming=True,
    ),

    # Required: Define how the agent accepts and returns data
    default_input_modes=["text"],
    default_output_modes=["text"],

    skills=[
        AgentSkill(
            id="create-refined-session",
            name="create_refined_session",
            description="Executes a 3-turn reflection loop to generate scripts.",
            tags=["synthesis", "meditation"],
        )
    ],

    metadata={
        "architecture": "LangGraph + A2A Protocol",
        "location": "Belgrade, Serbia",
        "developer_contact": "https://www.linkedin.com/in/bojanristic/"
    }
)
