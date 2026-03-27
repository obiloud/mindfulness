import logging
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from langgraph.runtime import Runtime
from ..state import SynthState, GraphContext, print_state
from ..prompts.meditation import ANSWER_PROMPT, MEDITATION_PROMPT

logger = logging.getLogger(__name__)


async def node_generate_answer(state: SynthState, runtime: Runtime[GraphContext]) -> dict:
    llm = runtime.context.llm

    logger.info(f"ANSWERING: {print_state(state)}")

    messages = state.get("messages", [])
    context_text = ""

    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if human_messages:
        context_text = "\n\n".join(m.content for m in human_messages)

    # Execute the search deterministically in Python
    search_tool = TavilySearch(max_results=2)
    search_query = f"Short mindfulness quote for someone feeling {context_text}"

    try:
        # Run the tool directly
        search_results = await search_tool.ainvoke({"query": search_query})
        # Extract just the content/snippets to save context window
        quotes_context = "\n".join([res for res in search_results])
    except Exception as e:
        logger.warning(f"Tavily search failed: {e}")
        quotes_context = "Fallback quote: 'Peace comes from within. Do not seek it without.' - Buddha"

    # Prepare context with the injected search results
    feedback = state.get("answer_feedback", "")

    # We update the prompt to include the quotes we just found
    prompt_context = ANSWER_PROMPT.format(
        user_query=context_text,
        transcript=state.get('transcript', '[Drafting...]')
    )

    prompt_context += f"\n\n# RETRIEVED QUOTES TO USE:\n{quotes_context}"

    if feedback:
        prompt_context += f"\n\n# ADDRESS THIS FEEDBACK: {feedback}"

    ai_msg = await llm.ainvoke([
        SystemMessage(content=MEDITATION_PROMPT),
        HumanMessage(content=prompt_context)
    ])

    return {
        "answer": ai_msg.content,
        "is_answer_valid": False  # Reset for re-validation
    }
