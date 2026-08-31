import os

import pytest

from langchain_learning_lab import credentials


def test_existing_environment_variable_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "already-present")
    monkeypatch.setattr(
        credentials,
        "_read_macos_keychain",
        lambda service: pytest.fail("Keychain should not be read"),
    )

    value = credentials.ensure_project_credential("DASHSCOPE_API_KEY")

    assert value == "already-present"


def test_keychain_value_is_added_only_to_current_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(
        credentials,
        "_read_macos_keychain",
        lambda service: "key-from-test-keychain",
    )

    value = credentials.ensure_project_credential("DASHSCOPE_API_KEY")

    assert value == "key-from-test-keychain"
    assert os.environ["DASHSCOPE_API_KEY"] == "key-from-test-keychain"


def test_project_outside_pythonproject_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "PROJECT_ROOT", tmp_path / "outside-project")
    monkeypatch.setattr(
        credentials,
        "_read_macos_keychain",
        lambda service: pytest.fail("Keychain should not be read"),
    )

    with pytest.raises(RuntimeError, match="outside PythonProject"):
        credentials.ensure_project_credential("DASHSCOPE_API_KEY")
