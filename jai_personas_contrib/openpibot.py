"""
Openpibot persona
"""

from __future__ import annotations

from typing import override

from .base_chat_persona import BaseChatPersona, BaseDiagnosticLLM


class OpenpibotLLM(BaseDiagnosticLLM):
    """Override LLM to add diagnostics on failure."""

    base_url_key: str = "OPENPIBOT_BASE_URL"
    api_key_key: str = "OPENPIBOT_API_KEY"


class OpenpibotPersona(BaseChatPersona):
    """Subclass of BaseChatPersona coded to connect to openpibot."""

    default_model_id: str = "openai/pi-agent"
    base_url_key: str = "OPENPIBOT_BASE_URL"
    base_url_default: str = "http://localhost:11435/v1"
    api_key_key: str = "OPENPIBOT_API_KEY"
    avatar_file: str = "pi.svg"
    persona_name: str = "Openpibot"
    llm_class: type[BaseDiagnosticLLM] = OpenpibotLLM

    @override
    async def get_tools(self) -> list:
        """Filter out the bash tool that causes errors with openpibot."""
        tools = await super().get_tools()
        return [tool for tool in tools if getattr(tool, "__name__", None) != "bash"]
