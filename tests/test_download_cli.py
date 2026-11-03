"""Model-prefetch CLI success and failure behavior."""

from click.testing import CliRunner

from ethnicolr2 import _setup_hooks


def test_download_cli_resolves_all_assets_and_reports_paths(monkeypatch):
    requested = []

    def resolve(filename):
        requested.append(filename)
        return f"/cache/{filename}"

    monkeypatch.setattr(_setup_hooks, "resolve_model", resolve)
    result = CliRunner().invoke(_setup_hooks.download_cli, ["--verbose"])
    assert result.exit_code == 0, result.output
    assert tuple(requested) == _setup_hooks.MODEL_FILES
    for filename in requested:
        assert f"{filename}: /cache/{filename}" in result.output
    assert f"{_setup_hooks.HF_REPO}@{_setup_hooks.HF_REVISION}" in result.output


def test_download_cli_quiet_output_and_failure(monkeypatch):
    monkeypatch.setattr(_setup_hooks, "resolve_model", lambda name: f"/cache/{name}")
    result = CliRunner().invoke(_setup_hooks.download_cli)
    assert result.exit_code == 0
    assert len(result.output.splitlines()) == 1
    assert "Resolved 6 assets" in result.output

    def unavailable(filename):
        raise OSError("offline")

    monkeypatch.setattr(_setup_hooks, "resolve_model", unavailable)
    result = CliRunner().invoke(_setup_hooks.download_cli)
    assert result.exit_code != 0
    assert isinstance(result.exception, OSError)
    assert "Resolved" not in result.output
