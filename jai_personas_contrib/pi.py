"""
pi.py - Persona for pi
"""

import os
import re
import shutil
import subprocess
from typing import override, ClassVar
from asyncio.subprocess import Process
from pathlib import Path

from acp.schema import (
    AgentMessageChunk,
)
from jupyter_ai_persona_manager import PersonaDefaults, PersonaRequirementsUnmet
from jupyter_ai_acp_client.base_acp_persona import BaseAcpPersona
from jupyter_ai_acp_client.default_acp_client import JaiAcpClient

# Path to the bundled pi.json that ships with this package.
# Configures permission: {edit: "ask", bash: "ask"} so Pi requests
# approval before file edits and shell commands.
_BUNDLED_CONFIG = os.path.join(os.path.dirname(__file__), "pi.json")


def _has_user_config() -> bool:
    """Check if user has a global Pi config file."""
    config_dir = Path.home() / ".config" / "pi"
    return (config_dir / "pi.json").exists() or (config_dir / "pi.jsonc").exists()


def _is_auth_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        keyword in message
        for keyword in (
            "api key",
            "api_key",
            "authentication",
            "authorized",
            "credential",
            "forbidden",
            "not configured",
        )
    )


def _check_pi() -> None:
    """Raise PersonaRequirementsUnmet if pi is missing or wrong version."""
    if shutil.which("pi") is None:
        raise PersonaRequirementsUnmet(
            "This persona requires `pi` to be installed."
            " See https://pi.ai for installation instructions."
        )

    try:
        result = subprocess.run(
            ["pi", "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except subprocess.TimeoutExpired as e:
        raise PersonaRequirementsUnmet(
            "pi --version command timed out."
            " Please ensure pi is properly installed."
        ) from e
    except FileNotFoundError as e:
        raise PersonaRequirementsUnmet(
            "pi command not found."
            " Please ensure pi is properly installed."
        ) from e

    if result.returncode != 0:
        stderr = result.stderr.strip()
        error_msg = (
            f"pi --version returned non-zero exit code {result.returncode}."
            " Please ensure pi is properly installed."
        )
        if stderr:
            error_msg += f"\nStderr output: {stderr}"
        raise PersonaRequirementsUnmet(error_msg)

    version_match = re.search(r"(\d+\.\d+\.\d+)", result.stdout)
    if not version_match:
        raise PersonaRequirementsUnmet(
            "Could not extract version number from pi --version output."
            f" Got: {result.stdout.strip()}"
        )

    version_str = version_match.group(1)
    current_version = tuple(int(x) for x in version_str.split("."))

    if current_version < (0, 70, 0):
        raise PersonaRequirementsUnmet(
            f"pi version {version_str} is installed,"
            " but version >=0.70.0 is required."
            " Please upgrade pi. See https://pi.ai for instructions."
        )


class PiAcpClient(JaiAcpClient):
    """
Acp client for Pi.  This ignores the startup message from the acp client.
    """
    def __init__(self, *args, **kwargs):
        _check_pi()
        super().__init__(*args, **kwargs)
        self.ignore_message = True

    @override
    def _handle_agent_message_chunk(self, session_id: str, update: AgentMessageChunk) -> None:
        """
This ignores the first message from pi-acp which is the startup message
        """
        if self.ignore_message:
            self.ignore_message = False
            return
        super()._handle_agent_message_chunk(session_id, update)


class PiAcpPersona(BaseAcpPersona):
    """
PiAcpPersona class
    """
    acp_client_class: ClassVar[type[JaiAcpClient]] = PiAcpClient
    def __init__(self, *args, **kwargs):
        executable = ["npx", "-y", "pi-acp"]
        super().__init__(*args, executable=executable, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "static", "pi.svg"
                )
            )
        )

        return PersonaDefaults(
            name="Pi",
            description="Pi as an ACP agent persona.",
            avatar_path=avatar_path,
            system_prompt="unused",
        )

    @override
    async def _init_agent_subprocess(
            self,
            env: dict[str, str] | None = None
    ) -> Process:
        if env is None:
            env = {}
        env["PI_ACP_ENABLE_EMBEDDED_CONTEXT"] = "true"

        # Only inject bundled config if the user hasn't configured Pi themselves.
        # Precedence: PI_CONFIG env var > ~/.config/pi/pi.{json,jsonc} > bundled
        if "PI_CONFIG" not in os.environ and not _has_user_config():
            env["PI_CONFIG"] = _BUNDLED_CONFIG

        return await super()._init_agent_subprocess(env=env)
