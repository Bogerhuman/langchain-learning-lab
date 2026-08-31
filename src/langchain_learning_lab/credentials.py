"""Load project-scoped model credentials from the macOS Keychain."""

import getpass
import os
import subprocess
import sys
from pathlib import Path


PYTHONPROJECT_ROOT = Path("/Users/xuchengbo/Documents/Code/PythonProject")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KEYCHAIN_SERVICES = {
    "DASHSCOPE_API_KEY": "codex-pythonproject-dashscope",
    "DEEPSEEK_API_KEY": "codex-pythonproject-deepseek",
}


def _read_macos_keychain(service: str) -> str:
    """Read one generic-password value without printing it to stdout or stderr."""
    if sys.platform != "darwin":
        raise RuntimeError("Automatic Keychain loading is supported only on macOS")

    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-w",
            "-a",
            getpass.getuser(),
            "-s",
            service,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        raise RuntimeError(
            f"Credential service {service!r} was not found in macOS Keychain"
        )
    return value


def ensure_project_credential(variable_name: str) -> str:
    """Ensure one credential exists only in this Python process's environment.

    An already supplied environment variable wins. Otherwise the credential is
    loaded only when this project's physical path is inside PythonProject.
    """
    existing_value = os.getenv(variable_name)
    if existing_value:
        return existing_value

    service = KEYCHAIN_SERVICES.get(variable_name)
    if service is None:
        raise ValueError(f"Unsupported credential variable: {variable_name}")

    project_root = PROJECT_ROOT.resolve()
    allowed_root = PYTHONPROJECT_ROOT.resolve()
    if not project_root.is_relative_to(allowed_root):
        raise RuntimeError(
            f"Refusing to load {variable_name}: project is outside PythonProject"
        )

    value = _read_macos_keychain(service)
    os.environ[variable_name] = value
    return value
