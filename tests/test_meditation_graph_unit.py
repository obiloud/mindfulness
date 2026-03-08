import pytest
import logging
from langchain_core.messages import AIMessage
import workflow
from state import GraphContext

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class FakeLLM:
    """Minimal fake LLM that returns deterministic AIMessage objects."""

    def __init__(self, response_text: str = "FAKE_ANSWER"):
        self.response_text = response_text

    def invoke(self, messages):
        return AIMessage(content=self.response_text)

@pytest.fixture
def fake_llm(monkeypatch):
    """Monkeypatch _get_llm to avoid external Hugging Face calls."""

    def _fake_get_llm():
        return FakeLLM()

    monkeypatch.setattr(workflow, "_get_llm", _fake_get_llm)


def test_run_mindfulness_graph_unsafe_input_refuses():
    """Safety detection should block clearly abusive language and not return a transcript."""
    result = workflow.run_mindfulness_graph("you are stupid")

    assert "refusal" in result["message"].lower() or "can't respond" in result["message"].lower()
    assert result["transcript"] is None


def test_run_mindfulness_graph_safe_input_allows(monkeypatch, fake_llm):
    """Benign input should not trigger the safety refusal path."""
    # Ensure run_mindfulness_graph sees our patched llm and transcript generator.
    result = workflow.run_mindfulness_graph("I feel a bit stressed about work lately.")

    assert result["message"]  # non-empty answer
    assert result["transcript"] == "FAKE_ANSWER"


def test_clarification_triggered_for_vague_short_query(monkeypatch):
    """Very short, generic queries should route through the clarification step."""

    # Use a simple LLM that echoes a clarifying question.
    class ClarificationLLM(FakeLLM):
        def invoke(self, messages):
            return AIMessage(content="Can you share a bit more about what is causing this feeling?")

    graph = workflow.build_mindfulness_graph()

    initial_state = {
        "query": "stress",
        "history": [],
        "messages": [],
        "transcript": None,
        "safety_flag": None,
        "refusal_message": None,
        "status": "initial",
        "clarification_count": 0,
        "reflection_count": 0,
    }

    dependencies = GraphContext(
        logger=logger,
        llm=ClarificationLLM()
    )

    final_state = graph.invoke(initial_state, context=dependencies)

    # After a vague query, the graph should move into a clarifying status and
    # produce at least one AIMessage asking a follow-up question.
    assert final_state["status"] in ("clarifying", "done")
    ai_messages = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
    assert ai_messages, "Expected at least one AIMessage from clarification node."
    assert "share a bit more" in ai_messages[-1].content


def test_router_does_not_prevent_completion(fake_llm):
    """End-to-end run should complete without hitting LangGraph recursion limits.

    This gives indirect confidence that the router function allows the workflow
    to reach a terminal state instead of looping forever.
    """

    result = workflow.run_mindfulness_graph("Tell me a bit about relaxation.")
    assert result["message"]
    assert result["transcript"] == "FAKE_ANSWER"

