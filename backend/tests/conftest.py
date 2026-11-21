import pytest


@pytest.fixture(autouse=True)
def configure_test_api_key(monkeypatch):
    """Provide an isolated API key for HTTP integration tests."""
    monkeypatch.setenv(
        "SKYGUARD_API_KEY",
        "test-only-secret-key-37450",
    )
