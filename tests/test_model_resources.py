"""Contracts for the published ethnicolr2 model assets."""

from unittest.mock import patch

import pytest

from ethnicolr2._resources import HF_REPO, HF_REVISION, resolve_model

MODEL_FILES = {
    "lstm_lastname_gen.pt",
    "lstm_fullname.pt",
    "census_lstm_lastname.pt",
    "pt_vec_lastname.joblib",
    "pt_vec_fullname.joblib",
    "pt_vec_census_lastname.joblib",
}


def test_explicit_model_directory_takes_precedence(tmp_path, monkeypatch):
    artifact = tmp_path / "lstm_fullname.pt"
    artifact.write_bytes(b"model")
    monkeypatch.setenv("ETHNICOLR2_MODEL_DIR", str(tmp_path))

    with patch("huggingface_hub.hf_hub_download") as download:
        assert resolve_model("models/lstm_fullname.pt") == str(artifact)
    download.assert_not_called()


def test_missing_model_uses_immutable_hub_revision(tmp_path, monkeypatch):
    monkeypatch.setenv("ETHNICOLR2_MODEL_DIR", str(tmp_path))

    with patch(
        "ethnicolr2._resources.hf_hub_download", return_value="/cache/model.pt"
    ) as download:
        assert resolve_model("models/not-bundled.pt") == "/cache/model.pt"
    download.assert_called_once_with(HF_REPO, "not-bundled.pt", revision=HF_REVISION)


def test_revision_is_an_immutable_commit():
    assert len(HF_REVISION) == 40
    assert set(HF_REVISION) <= set("0123456789abcdef")


@pytest.mark.live
def test_pinned_revision_contains_every_model_file():
    from huggingface_hub import list_repo_files

    published = set(list_repo_files(HF_REPO, revision=HF_REVISION))
    assert MODEL_FILES <= published


@pytest.mark.live
def test_published_vectorizers_match_installed_scikit_learn():
    import warnings

    import joblib
    from huggingface_hub import hf_hub_download
    from sklearn.exceptions import InconsistentVersionWarning

    vectorizers = sorted(name for name in MODEL_FILES if name.endswith(".joblib"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = [
            joblib.load(hf_hub_download(HF_REPO, name, revision=HF_REVISION))
            for name in vectorizers
        ]

    assert all(vectorizer.vocabulary_ for vectorizer in loaded)
    assert not any(
        isinstance(item.message, InconsistentVersionWarning) for item in caught
    )
