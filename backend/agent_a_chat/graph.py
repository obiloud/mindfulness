# agent_a_chat/graph.py
import inspect
from typing import Literal
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from .nodes.node_user_input_agent import node_user_input
from .nodes.node_conversation_agent import node_conversation
from .nodes.node_proactive_engagement import node_proactive_engagement
from .nodes.node_hydrate import node_hydrate
from .nodes.node_manage_memory import node_manage_memory
from .nodes.node_deliver_transcript import node_deliver_transcript
from .state import ChatState, GraphContext


def should_trigger_synth(state: ChatState) -> ChatState:
    """Evaluates the patience loop threshold."""
    status = state.get("synth_status", "idle")

    if status == "idle" and (state.get("info_score", 0) >= 0.8 or state.get("turn_count", 0) > 5):
        return {"synth_status": "requested"}

    return {}


def create_chat_graph(checkpointer: BaseCheckpointSaver = None, store: BaseStore = None):
    """
    Builds the fast Chat Graph.
    Evaluates context and flags when to trigger the Synthesis Agent.
    """

    workflow = StateGraph(ChatState, context_schema=GraphContext)

    # We pass llm to nodes via partials or adjust your node implementations to accept it
    workflow.add_node("user", node_user_input)
    workflow.add_node("hydrate", node_hydrate)
    workflow.add_node("conversation", node_conversation)
    workflow.add_node("manage_memory", node_manage_memory)
    workflow.add_node("evaluate_patience", should_trigger_synth)
    workflow.add_node("proactive_engagement", node_proactive_engagement)
    workflow.add_node("deliver_transcript", node_deliver_transcript)

    workflow.set_entry_point("user")

    async def route_after_user(state: ChatState, runtime: Runtime[GraphContext]) -> Literal["safe_to_proceed", "deliver_transcript", "end"]:
        if state.get("safety_flag") == "unsafe":
            return "end"
        # If we were waiting for a confirmation, check if they said "Yes"
        if state.get("awaiting_confirmation"):
            llm = runtime.context.llm
            user_input = state["messages"][-1].content.lower()

            intent_prompt = ChatPromptTemplate.from_template(inspect.cleandoc("""
                The user was asked if they want to start their mindfulness session. 
                Based on their reply, do they want to proceed? 
                Reply 'START' to proceed or 'WAIT' to stay in conversation.

                User: {text}
            """))

            response = await (intent_prompt | llm).ainvoke({"text": user_input})
            if "START" in response.content.upper():
                return "deliver_transcript"

        return "safe_to_proceed"

    workflow.add_conditional_edges("user", route_after_user, {
        "deliver_transcript": "deliver_transcript",
        "safe_to_proceed": "hydrate",
        "end": END
    })

    workflow.add_edge("hydrate", "conversation")
    workflow.add_edge("conversation", "manage_memory")

    def route_after_conversation(state: ChatState) -> Literal["proactive_engagement", "evaluate_patience"]:
        if state.get("is_synthesis_ready") and not state.get("awaiting_confirmation"):
            return "proactive_engagement"
        return "evaluate_patience"

    workflow.add_conditional_edges(
        "manage_memory",
        route_after_conversation,
        {
            "proactive_engagement": "proactive_engagement",
            "evaluate_patience": "evaluate_patience"
        })

    workflow.add_edge("deliver_transcript", "evaluate_patience")
    workflow.add_edge("proactive_engagement", "evaluate_patience")
    workflow.add_edge("evaluate_patience", END)

    return workflow.compile(checkpointer=checkpointer, store=store)
