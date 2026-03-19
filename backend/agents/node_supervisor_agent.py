from state import ConversationState, GraphContext, print_state
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.runtime import Runtime
from prompts.supervisor import SUPERVISOR_PROMPT


async def node_reflection(state: ConversationState, runtime: Runtime[GraphContext]) -> ConversationState:
    """
    Iterative reflection: the model critiques and, if needed, refines
    its previous answer into a clearer, more soothing response.
    """

    logger = runtime.context.logger
    llm = runtime.context.llm

    logger.info(f"REFLECTING ON STATE: {print_state(state)}")

    # Handle edge case: Missing content
    if not state.get("transcript") or len(state["transcript"].strip()) < 10:
        return {
            "transcript_feedback": "Transcript was empty. Generate a full 10-minute meditation.",
            "is_transcript_valid": False,
            "status": "reflecting"
        }

    # Invoke Supervisor
    reflection_prompt = SUPERVISOR_PROMPT.format(
        answer=state.get('answer', ''),
        transcript=state.get('transcript', '')
    )

    # Using aconcise call to save tokens
    response = await llm.ainvoke([SystemMessage(content=reflection_prompt)])
    res_text = response.content if isinstance(
        response, AIMessage) else str(response)

    # 3. Deterministic Parsing (Bypassing Pydantic issues)
    # We parse the two statuses independently
    ans_part = [line for line in res_text.split(
        '\n') if "ANSWER_STATUS:" in line]
    tra_part = [line for line in res_text.split(
        '\n') if "TRANSCRIPT_STATUS:" in line]

    ans_status = ans_part[0].replace(
        "ANSWER_STATUS:", "").strip() if ans_part else "FEEDBACK: Retry"
    tra_status = tra_part[0].replace(
        "TRANSCRIPT_STATUS:", "").strip() if tra_part else "FEEDBACK: Retry"

    # 4. Evaluate Success
    is_ans_valid = ans_status.upper() == "SATISFACTORY"
    is_tra_valid = tra_status.upper() == "SATISFACTORY"

    # 5. Handle Reflection Budget
    current_count = state.get("reflection_count", 0)
    max_reflections = 1

    if current_count >= max_reflections:
        is_ans_valid = True
        is_tra_valid = True

    # 6. Extract Feedback if not satisfactory
    def get_feedback(status_str):
        if "FEEDBACK:" in status_str:
            return status_str.split("FEEDBACK:", 1)[1].strip()
        return None

    return {
        # **state,
        "is_answer_valid": is_ans_valid,
        "is_transcript_valid": is_tra_valid,
        "answer_feedback": get_feedback(ans_status) if not is_ans_valid else None,
        "transcript_feedback": get_feedback(tra_status) if not is_tra_valid else None,
        "reflection_count": current_count + 1,
        "status": "done" if (is_ans_valid and is_tra_valid) else "reflecting"
    }
