"""
Openclaw persona
"""

from __future__ import annotations

from typing import override

from .base_chat_persona import BaseChatPersona, BaseDiagnosticLLM


class OpenclawLLM(BaseDiagnosticLLM):
    """Override LLM to add diagnostics on failure."""

    base_url_key: str = "OPENCLAW_BASE_URL"
    api_key_key: str = "OPENCLAW_API_KEY"


class OpenclawPersona(BaseChatPersona):
    """Subclass of BaseChatPersona coded to connect to openclaw."""

    default_model_id: str = "openai/openclaw"
    base_url_key: str = "OPENCLAW_BASE_URL"
    base_url_default: str = "http://localhost:18789/v1"
    api_key_key: str = "OPENCLAW_API_KEY"
    avatar_file: str = "openclaw.svg"
    persona_name: str = "Openclaw"
    llm_class: type[BaseDiagnosticLLM] = OpenclawLLM

    @override
    async def get_tools(self) -> list:
        """Filter out the bash tool that causes errors with openclaw."""
        tools = await super().get_tools()
        return [tool for tool in tools if getattr(tool, "__name__", None) != "bash"]
