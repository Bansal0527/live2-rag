import pytest
from langchain_core.runnables import RunnableLambda
from app.llm import GeminiService, InvalidModelOutput
from app.parsing import PageText
from app.schemas import Source


class FakeStructuredModel:
    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def with_structured_output(self, schema):
        return RunnableLambda(lambda _: next(self.outputs))


def test_extraction_retries_unsupported_quote():
    bad = {"document_type": "Agreement", "facts": [{
        "name": "notice", "value": "60 days", "evidence": "90 days", "page": 1,
    }]}
    good = {"document_type": "Agreement", "facts": [{
        "name": "notice", "value": "60 days", "evidence": "60 days", "page": 1,
    }]}
    ai = GeminiService(FakeStructuredModel([bad, good]))
    result = ai.extract("agreement.pdf", [PageText(1, "The notice period is 60 days.")])
    assert result.facts[0].value == "60 days"


def test_extraction_fails_after_two_invalid_responses():
    ai = GeminiService(FakeStructuredModel([{"facts": [{"name": "x"}]}, {"facts": [{"name": "x"}]}]))
    with pytest.raises(InvalidModelOutput):
        ai.extract("agreement.txt", [PageText(None, "Some content")])


def test_extraction_preserves_conflicting_facts_and_missing_type():
    ai = GeminiService(FakeStructuredModel([{
        "document_type": None,
        "facts": [
            {"name": "claim deadline", "value": "10 days", "evidence": "claims within 10 days", "page": None},
            {"name": "claim deadline", "value": "20 days", "evidence": "claims within 20 days", "page": None},
        ],
    }]))
    result = ai.extract("draft.txt", [PageText(None, "Draft says claims within 10 days. Another section says claims within 20 days.")])
    assert result.document_type is None
    assert [fact.value for fact in result.facts] == ["10 days", "20 days"]


def test_answer_agent_uses_retrieval_tool(monkeypatch):
    source = Source(document_id="doc", document_name="test.txt", page=None,
                    chunk_id="doc:1", text="Notice is 60 days.", similarity=0.9)

    class Agent:
        def invoke(self, _):
            tool_result = tools["search_documents"].invoke({"search_query": "notice"})
            assert "doc:1" in tool_result
            metadata_result = tools["list_document_metadata"].invoke({"document_id": "doc"})
            assert '"id": "doc"' in metadata_result
            return {"structured_response": {"answer": "60 days", "citation_ids": ["doc:1"]}}

    def fake_create_agent(**kwargs):
        nonlocal tools
        tools = {item.name: item for item in kwargs["tools"]}
        return Agent()

    tools = {}
    monkeypatch.setattr("app.llm.create_agent", fake_create_agent)
    ai = GeminiService(FakeStructuredModel([]))
    answer, sources = ai.answer("How much notice?", lambda _: [source], lambda _: [{"id": "doc"}])
    assert answer.answer == "60 days"
    assert sources == [source]


def test_invalid_agent_answer_abstains(monkeypatch):
    class Agent:
        def invoke(self, _):
            raise ValueError("Malformed structured response")

    monkeypatch.setattr("app.llm.create_agent", lambda **_: Agent())
    ai = GeminiService(FakeStructuredModel([]))
    answer, sources = ai.answer("Unknown value?", lambda _: [], lambda _: [])
    assert answer.answer is None and sources == []
