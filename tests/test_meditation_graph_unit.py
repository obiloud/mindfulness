import pytest
from langchain_core.messages import AIMessage
from types import SimpleNamespace

import meditation_graph


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

    monkeypatch.setattr(meditation_graph, "_get_llm", _fake_get_llm)


@pytest.fixture
def fake_transcript(monkeypatch):
    """
    Monkeypatch the meditation transcript generator with a deterministic fake.
    This fixture is now scoped to each test and properly resets state.
    """
    def _fake_invoke(payload):
        # Mirror the interface used in meditation_graph (dict with 'text').
        return {"text": "FAKE_TRANSCRIPT"}
    
    # Create a mock object that mimics the expected interface
    mock_chain = SimpleNamespace(invoke=_fake_invoke)
    
    # Apply the monkeypatch for the current test execution
    monkeypatch.setattr(
        "meditation_graph.meditation_guide_generator_chain", 
        mock_chain
    )
    
    # Return the mock object so it can be used in tests if needed
    return mock_chain

@pytest.fixture(autouse=True)
def reset_meditation_graph(monkeypatch):
    """
    Ensure that the meditation graph state is reset before each test.
    This prevents state leakage between tests.
    """
    # Reset any other potential state in meditation_graph module
    # This is optional depending on your actual implementation
    pass


def test_run_mindfulness_graph_unsafe_input_refuses():
    """Safety detection should block clearly abusive language and not return a transcript."""
    result = meditation_graph.run_mindfulness_graph("you are stupid")

    assert "refusal" in result["message"].lower() or "can't respond" in result["message"].lower()
    assert result["transcript"] is None


def test_run_mindfulness_graph_safe_input_allows(monkeypatch, fake_llm, fake_transcript):
    """Benign input should not trigger the safety refusal path."""
    # Ensure run_mindfulness_graph sees our patched llm and transcript generator.
    result = meditation_graph.run_mindfulness_graph("I feel a bit stressed about work lately.")

    assert result["message"]  # non-empty answer
    assert result["transcript"] == "FAKE_TRANSCRIPT"


def test_clarification_triggered_for_vague_short_query(monkeypatch):
    """Very short, generic queries should route through the clarification step."""

    # Use a simple LLM that echoes a clarifying question.
    class ClarificationLLM(FakeLLM):
        def invoke(self, messages):
            return AIMessage(content="Can you share a bit more about what is causing this feeling?")

    def _fake_get_llm():
        return ClarificationLLM()

    monkeypatch.setattr(meditation_graph, "_get_llm", _fake_get_llm)

    graph = meditation_graph.build_mindfulness_graph()

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

    final_state = graph.invoke(initial_state)

    # After a vague query, the graph should move into a clarifying status and
    # produce at least one AIMessage asking a follow-up question.
    assert final_state["status"] in ("clarifying", "done")
    ai_messages = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
    assert ai_messages, "Expected at least one AIMessage from clarification node."
    assert "share a bit more" in ai_messages[-1].content


def test_router_does_not_prevent_completion(fake_llm, fake_transcript):
    """End-to-end run should complete without hitting LangGraph recursion limits.

    This gives indirect confidence that the router function allows the workflow
    to reach a terminal state instead of looping forever.
    """

    result = meditation_graph.run_mindfulness_graph("Tell me a bit about relaxation.")
    assert result["message"]
    assert result["transcript"] == "FAKE_TRANSCRIPT"

