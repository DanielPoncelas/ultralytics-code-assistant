import numpy as np

from code_assistant.embeddings import EMBEDDING_MODELS
from code_assistant.indexing import embedding_cache

SPEC = EMBEDDING_MODELS["BAAI/bge-small-en-v1.5"]


def test_fingerprint_changes_with_text_and_model():
    base = embedding_cache.fingerprint(["a", "b"], SPEC)
    other_model = EMBEDDING_MODELS["Qwen/Qwen3-Embedding-0.6B"]

    assert base == embedding_cache.fingerprint(["a", "b"], SPEC)
    assert base != embedding_cache.fingerprint(["a", "c"], SPEC)
    assert base != embedding_cache.fingerprint(["b", "a"], SPEC)  # order matters
    assert base != embedding_cache.fingerprint(["a", "b"], other_model)


def test_save_and_load_round_trip(tmp_path):
    path = embedding_cache.cache_path(tmp_path, SPEC, "abc")
    vectors = np.random.default_rng(0).random((2, 4), dtype=np.float32)

    embedding_cache.save(path, ["id1", "id2"], vectors)

    np.testing.assert_array_equal(embedding_cache.load(path, ["id1", "id2"]), vectors)


def test_load_rejects_missing_file_or_different_ids(tmp_path):
    path = embedding_cache.cache_path(tmp_path, SPEC, "abc")
    assert embedding_cache.load(path, ["id1"]) is None

    embedding_cache.save(path, ["id1"], np.zeros((1, 4), dtype=np.float32))
    assert embedding_cache.load(path, ["other"]) is None


def test_load_ignores_a_corrupt_cache_file(tmp_path):
    path = embedding_cache.cache_path(tmp_path, SPEC, "abc")
    path.write_bytes(b"truncated")

    assert embedding_cache.load(path, ["id1"]) is None
