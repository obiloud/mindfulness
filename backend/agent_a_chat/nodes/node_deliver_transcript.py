from ..state import ChatState
from langchain_core.messages import AIMessage


async def node_deliver_transcript(state: ChatState) -> ChatState:
    """
    Prepares the final transcript with chapterized content for delivery.
    Adds chapterized transcript to additional_kwargs for the final message.
    """
    transcript = state.get("transcript", "")
    chapters = state.get("chapters", [])

    # Build additional_kwargs with chapterized transcript
    additional_kwargs = {
        "answer": state.get("answer", ""),
        "transcript": transcript,
        "chapters": chapters
    }

    return {
        "messages": [
            AIMessage(
                content="Excellent. Let's begin.",
                additional_kwargs=additional_kwargs
            )
        ],
        "awaiting_confirmation": False,
        "is_synthesis_ready": False  # Reset the flag
    }
