from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from state import ConversationState, GraphContext, print_state
from langgraph.runtime import Runtime
from prompts.meditation import MEDITATION_PROMPT, ANSWER_PROMPT, TRANSCRIPT_PROMPT


def node_assistant(state: ConversationState, runtime: Runtime[GraphContext]) -> ConversationState:
    """Main assistant step that generates both answer and transcript."""
    logger = runtime.context.logger
    llm = runtime.context.llm

    logger.debug(
        f"Response node: generating transcript for state='{print_state(state)}'")

    messages = state["messages"]

    # Decide whether this is the initial pass or a refinement loop.
    reflection_notes = state.get("reflection_notes")
    is_refinement = bool(reflection_notes)

    # Ensure we have a meditation transcript, generating it directly in code.

    # Build a simple textual context from the conversation.
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if human_messages:
        context_text = "\n\n".join(m.content for m in human_messages)
    else:
        context_text = state.get("query", "")

    if is_refinement:
        context_text = f"{context_text}\n\n# IMPORTANT: Please address this feedback: {reflection_notes}"

    try:
        transcript_agent_system = SystemMessage(content=TRANSCRIPT_PROMPT)
        transcript_agent_context = HumanMessage(
            content=f"""Context for the guided session: {context_text}""")
        raw_transcript = llm.invoke(
            [transcript_agent_system, transcript_agent_context])
        transcript = raw_transcript.content if isinstance(
            raw_transcript, AIMessage) else str(raw_transcript)
    except Exception:
        # In case transcript generation fails, fall back to an empty string.
        transcript = ""

    system_prompt = MEDITATION_PROMPT

    if reflection_notes:
        system_prompt += f"\n\n# IMPORTANT: Please address this feedback when refining your answer: {reflection_notes}"

    system = SystemMessage(content=system_prompt)

    answer_context = ANSWER_PROMPT.format(
        user_query=state.get('query', ''),
        transcript=transcript
    )

    answer_human = HumanMessage(
        content=answer_context
    )

    ai_msg = llm.invoke([system] + messages + [answer_human])
    answer_text = ai_msg.content if isinstance(
        ai_msg, AIMessage) else str(ai_msg)

    return {
        **state,
        "answer": answer_text,
        "transcript": transcript,
        "messages": [ai_msg],
        "status": "answering" if not is_refinement else state.get("status", "answering"),
    }
