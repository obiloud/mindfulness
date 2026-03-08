from state import ConversationState, GraphContext, print_state
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.runtime import Runtime
from prompts.supervisor import SUPERVISOR_PROMPT

def node_reflection(state: ConversationState, runtime: Runtime[GraphContext]) -> ConversationState:
    """
    Iterative reflection: the model critiques and, if needed, refines
    its previous answer into a clearer, more soothing response.
    """

    logger = runtime.context.logger
    llm = runtime.context.llm

    logger.info(f"Reflection node: reflecting on last respone state='{print_state(state)}'")

    if not state.get("transcript") or len(state["transcript"].strip()) < 10:
        return {
            **state,
            "reflection_notes": "The assistant failed to generate a meditation transcript. Please provide a full script.",
            "status": "reflecting"
        }

    messages = state["messages"]
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)

    if last_ai is None:
        return {
            **state,
            "status": "done",
        }
    
    reflection_prompt = SUPERVISOR_PROMPT.format(
        answer = state.get('answer', ''),
        transcript = state.get('transcript', '')
    )

    reflect_msg = SystemMessage(content=reflection_prompt)
    reflect_ai = llm.invoke([reflect_msg] + messages)
    reflect_text = reflect_ai.content.strip() if isinstance(reflect_ai, AIMessage) else str(reflect_ai).strip()

    logger.info(f"REFLECT TEXT: {reflect_text}\n")

    max_reflections = 3
    current_count = state.get("reflection_count", 0)

    lower = reflect_text.lower()
    if lower.startswith("satisfactory") or current_count >= max_reflections:
        return {
            **state,
            "reflection_count": current_count,
            "status": "done",
        }

    feedback = reflect_text
    if ":" in reflect_text:
        feedback = reflect_text.split(":", 1)[1].strip() or reflect_text

    return {
        **state,
        "messages": messages,
        "status": "reflecting",
        "reflection_count": current_count + 1,
        "reflection_notes": feedback,
    }