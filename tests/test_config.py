import pytest
from pydantic import ValidationError

from code_assistant.config import Settings

URI = "mongodb+srv://user:supersecret@cluster.example.mongodb.net/"
KEY = "sk-or-v1-supersecret"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Run each test with no real `.env` and no inherited credentials."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def test_loads_from_environment(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", URI)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)

    settings = Settings()

    assert settings.mongodb_uri.get_secret_value() == URI
    assert settings.openrouter_api_key.get_secret_value() == KEY
    assert settings.mongodb_database == "code_assistant"


def test_loads_from_dotenv_file(tmp_path):
    (tmp_path / ".env").write_text(f"MONGODB_URI={URI}\nOPENROUTER_API_KEY={KEY}\n")

    settings = Settings()

    assert settings.openrouter_api_key.get_secret_value() == KEY


def test_missing_variables_fail_with_their_names():
    with pytest.raises(ValidationError) as exc_info:
        Settings()

    missing = {err["loc"][0] for err in exc_info.value.errors()}
    assert missing == {"mongodb_uri", "openrouter_api_key"}


def test_secrets_are_not_exposed_in_repr(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", URI)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)

    rendered = repr(Settings())

    assert "supersecret" not in rendered


def test_llm_models_can_be_given_as_comma_separated_env_var(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", URI)
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    monkeypatch.setenv("LLM_MODELS", "a/model:free, b/model:free")

    assert Settings().llm_models == ("a/model:free", "b/model:free")


def test_each_embedding_model_gets_its_own_collection():
    settings = Settings(mongodb_uri=URI, openrouter_api_key=KEY)

    assert settings.collection_for("BAAI/bge-small-en-v1.5") == "chunks"
    assert settings.collection_for("qwen/qwen3-embedding-8b") == "chunks_qwen3-embedding-8b"
    assert settings.indexed_models == ["BAAI/bge-small-en-v1.5", "qwen/qwen3-embedding-8b"]
    alone = Settings(mongodb_uri=URI, openrouter_api_key=KEY, alt_embedding_model="")
    assert alone.indexed_models == ["BAAI/bge-small-en-v1.5"]
