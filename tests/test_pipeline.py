import pytest

from code_assistant.indexing.models import Chunk
from code_assistant.indexing.pipeline import SourceInfo, build_document, path_metadata
from code_assistant.store import vector_index_definition

SOURCE = SourceInfo("https://github.com/ultralytics/ultralytics", "v8.4.175", "abc123")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "ultralytics/models/yolo/segment/train.py",
            {"folder": "models", "model_family": "yolo", "task": "segment"},
        ),
        (
            "ultralytics/models/yolo/model.py",
            {"folder": "models", "model_family": "yolo", "task": None},
        ),
        (
            "ultralytics/models/sam/predict.py",
            {"folder": "models", "model_family": "sam", "task": None},
        ),
        (
            "ultralytics/models/__init__.py",
            {"folder": "models", "model_family": None, "task": None},
        ),
        (
            "ultralytics/engine/trainer.py",
            {"folder": "engine", "model_family": None, "task": None},
        ),
    ],
)
def test_path_metadata(path, expected):
    assert path_metadata(path) == expected


def test_build_document_has_permalink_and_provenance():
    chunk = Chunk(
        kind="method",
        path="ultralytics/engine/trainer.py",
        qualified_name="BaseTrainer.save_model",
        code="def save_model(self): ...",
        start_line=10,
        end_line=20,
        parent_class="BaseTrainer",
    )

    doc = build_document(chunk, [0.1, 0.2], SOURCE, "some/model")

    assert doc["_id"] == chunk.id
    assert doc["github_url"] == (
        "https://github.com/ultralytics/ultralytics/blob/abc123/"
        "ultralytics/engine/trainer.py#L10-L20"
    )
    assert doc["source"] == {"repo": SOURCE.repo_url, "tag": "v8.4.175", "commit": "abc123"}
    assert doc["embedding_model"] == "some/model"
    assert doc["embedding"] == [0.1, 0.2]
    assert doc["folder"] == "engine"


def test_vector_index_definition_declares_vector_and_filter_fields():
    definition = vector_index_definition(1024)

    vector, *filters = definition["fields"]
    assert vector == {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1024,
        "similarity": "cosine",
    }
    assert {f["path"] for f in filters} == {"folder", "kind", "model_family", "task"}
    assert all(f["type"] == "filter" for f in filters)
