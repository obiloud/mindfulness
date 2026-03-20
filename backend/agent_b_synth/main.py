# agent_b_synth/main.py
from a2a.server import A2AServer
from a2a.types import AgentCard, Skill, Parameter
from graph import synth_graph  # Your existing 3-turn reflection graph

synth_skill = Skill(
    name="generate_meditation",
    description="Generates a high-quality mindfulness transcript via reflection",
    parameters=[Parameter(name="context", type="string")]
)


class SynthesizerAgent(A2AServer):
    async def on_task_start(self, task):
        # The 'context' contains the summary or state from Agent A
        context = task.input_data["context"]

        # Invoke the LangGraph logic
        result = await synth_graph.ainvoke({"input_context": context})

        # Return the generated artifact
        return {"transcript": result["final_transcript"]}


# card = AgentCard(agent_id="pulse-synth-v1", skills=[synth_skill])
