import subprocess
import sys
from pathlib import Path

from src.generator import DEFAULT_MODEL, REFUSAL_ANSWER, format_context_chunks, generate
from src.schema import Chunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    def __init__(self, answer: str = "Grounded answer.") -> None:
        self.chat = FakeChat(answer)


class FakeChat:
    def __init__(self, answer: str) -> None:
        self.completions = FakeCompletions(answer)


class FakeCompletions:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self.answer)


class FakeResponse:
    def __init__(self, answer: str) -> None:
        self.choices = [FakeChoice(answer)]


class FakeChoice:
    def __init__(self, answer: str) -> None:
        self.message = FakeMessage(answer)


class FakeMessage:
    def __init__(self, answer: str) -> None:
        self.content = answer


def test_format_context_chunks_includes_citation_docstring_and_source() -> None:
    text = format_context_chunks([_chunk()])

    assert "src/app.py::run lines 10-12" in text
    assert "Docstring: Run the app." in text
    assert "def run" in text


def test_format_context_chunks_accepts_dict_chunks() -> None:
    text = format_context_chunks([_chunk().to_dict()])

    assert "src/app.py::run lines 10-12" in text


def test_generate_returns_refusal_for_empty_context_without_calling_client() -> None:
    client = FakeClient()

    answer = generate([], "What does this do?", client=client)

    assert answer == REFUSAL_ANSWER
    assert client.chat.completions.calls == []


def test_generate_passes_grounded_messages_to_client() -> None:
    client = FakeClient(answer="run returns 1. [1]")

    answer = generate([_chunk()], "What does run do?", temperature=0.2, client=client)

    assert answer == "run returns 1. [1]"
    [call] = client.chat.completions.calls
    assert call["model"] == DEFAULT_MODEL
    assert call["temperature"] == 0.2
    messages = call["messages"]
    assert messages[0]["role"] == "system"
    assert "strictly using the provided source code snippets" in messages[0]["content"]
    assert "Every factual sentence" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "What does run do?" in messages[1]["content"]
    assert "src/app.py::run lines 10-12" in messages[1]["content"]
    assert "Use snippet citations" in messages[1]["content"]


def test_generate_accepts_dict_chunks() -> None:
    client = FakeClient(answer="dict answer [1]")

    answer = generate([_chunk().to_dict()], "What does run do?", client=client)

    assert answer == "dict answer [1]"


def test_generate_uses_refusal_for_blank_model_response() -> None:
    client = FakeClient(answer="   ")

    answer = generate([_chunk()], "What does run do?", client=client)

    assert answer == REFUSAL_ANSWER


def test_generate_uses_refusal_for_uncited_model_response() -> None:
    client = FakeClient(answer="run returns 1.")

    answer = generate([_chunk()], "What does run do?", client=client)

    assert answer == REFUSAL_ANSWER


def test_generate_refuses_when_answer_omits_code_identifier_from_query() -> None:
    client = FakeClient(answer="The handler returns validation errors. [1]")

    answer = generate(
        [_chunk().to_dict() | {"function_name": "RequestValidationError"}],
        "Where is RequestValidationError handled?",
        client=client,
    )

    assert answer == REFUSAL_ANSWER


def test_generate_allows_answer_that_mentions_code_identifier_from_query() -> None:
    client = FakeClient(answer="RequestValidationError is handled here. [1]")

    answer = generate(
        [_chunk().to_dict() | {"function_name": "RequestValidationError"}],
        "Where is RequestValidationError handled?",
        client=client,
    )

    assert answer == "RequestValidationError is handled here. [1]"


def test_cli_help_is_available_without_ollama() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.generator", "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Generate a grounded answer" in result.stdout


def _chunk() -> Chunk:
    return Chunk(
        id="run",
        filepath="src/app.py",
        function_name="run",
        start_line=10,
        end_line=12,
        docstring="Run the app.",
        source="def run():\n    return 1\n",
        language="python",
    )
