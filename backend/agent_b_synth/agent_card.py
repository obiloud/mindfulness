from a2a.types import AgentCard, Skill, Parameter

# The 'Producer' Agent Identity
PULSE_SYNTH_CARD = AgentCard(
    agent_id="pulse-lotus-synth-v1",
    name="Pulse Lotus: Meditation Synthesizer",
    description="A high-fidelity reflection agent specialized in multi-pass mindfulness transcript generation.",

    # --- Professional Metadata for Portfolio ---
    metadata={
        "version": "1.0.0",
        "developer": "Bojan Ristic",
        "organization": "Mad Grap Studio Beograd",
        "entity": "Bojan Ristic PR, Agencija za racunarsko programiranje",
        "location": "Belgrade, Serbia",
        "architecture": "LangGraph + A2A Protocol",
        "specialization": "Generative AI & LLM Orchestration",
        "contact": "https://opsly.com"  # Or your professional portfolio URL
    },

    skills=[
        Skill(
            name="create_refined_session",
            description="Executes a 3-turn reflection loop to generate and validate a personalized meditation script.",
            parameters=[
                Parameter(
                    name="user_intent",
                    type="string",
                    description="The raw emotional or situational context from the user."
                ),
                Parameter(
                    name="reflection_depth",
                    type="integer",
                    default=3,
                    description="Number of self-correction passes to perform."
                )
            ],
            output_schema={
                "transcript": "string",
                "reflection_log": "list",
                "quality_score": "float"
            }
        )
    ]
)
