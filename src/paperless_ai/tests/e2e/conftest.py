import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("e2e-demo")
    group.addoption(
        "--real-llm",
        action="store_true",
        default=False,
        help="Call a real OpenAI-compatible LLM instead of the scripted model.",
    )
    group.addoption(
        "--llm-endpoint",
        default="http://bb-framework:8080/v1",
        help="OpenAI-compatible endpoint for --real-llm.",
    )
    group.addoption(
        "--llm-model",
        default="unsloth-Qwen3.6-35B-A3B",
        help="Model name for --real-llm.",
    )


@pytest.fixture
def real_llm(request: pytest.FixtureRequest) -> dict | None:
    if not request.config.getoption("--real-llm"):
        return None
    return {
        "endpoint": request.config.getoption("--llm-endpoint"),
        "model": request.config.getoption("--llm-model"),
    }
