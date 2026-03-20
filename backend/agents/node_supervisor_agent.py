from state import ConversationState, GraphContext, print_state
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime
from prompts.supervisor import SUPERVISOR_PROMPT
from pydantic import BaseModel, Field
import json
from functools import reduce


class ReflectionOutput(BaseModel):
    is_answer_valid: str = Field("Whether the answer meets all constraints")
    is_transcript_valid: str = Field(
        "Whether the transcript meets all constraints")
    answer_feedback: str = Field(
        description="Detailed feedback for the answer, including specific areas for improvement")
    transcript_feedback: str = Field(
        description="Detailed feedback for the transcript, including specific areas for improvement")


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

    structured_llm = llm.with_structured_output(
        ReflectionOutput, method="json_schema")
    reflection = await structured_llm.ainvoke([SystemMessage(content=reflection_prompt)])

    logger.info(f"reflection: {json.dumps(reflection, indent=2)}")

    if isinstance(reflection, list):
        reflection = reflection[0]

    if not isinstance(reflection, dict):
        reflection = {
            "is_answer_valid": False,
            "is_transcript_valid": False,
            "answer_feedback": "try again",
            "transcript_feedback": "try again"
        }

    is_ans_valid = reflection.get("is_answer_valid")
    is_tra_valid = reflection.get("is_transcript_valid")

    current_count = state.get("reflection_count", 0)
    max_reflections = 3

    if current_count >= max_reflections:
        is_ans_valid = True
        is_tra_valid = True

    return {
        "is_answer_valid": is_ans_valid,
        "is_transcript_valid": is_tra_valid,
        "answer_feedback": "\n".join(reflection.get("answer_feedback")) if not is_ans_valid else None,
        "transcript_feedback": "\n".join(reflection.get("transcript_feedback")) if not is_ans_valid else None,
        "reflection_count": current_count + 1,
        "status": "done" if (is_ans_valid and is_tra_valid) else "reflecting"
    }
