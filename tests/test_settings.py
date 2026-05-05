from pathlib import Path

from severewx.config import load_settings


def test_env_config_overrides_working_directory_config(tmp_path: Path, monkeypatch) -> None:
    local_config = tmp_path / "config.yaml"
    env_config = tmp_path / "ci.yaml"
    local_config.write_text("ingest:\n  source: local_staged_gfs\n", encoding="utf-8")
    env_config.write_text("ingest:\n  source: nomads\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SEVEREWX_CONFIG", str(env_config))

    settings = load_settings()

    assert settings.get("ingest.source") == "nomads"
