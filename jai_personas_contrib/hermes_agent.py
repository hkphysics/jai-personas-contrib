"""
hermes_agent - Persona for Hermes Agent
"""

from __future__ import annotations

from typing import override

from .base_chat_persona import BaseChatPersona, BaseDiagnosticLLM


def _remove_reasoning_objects(obj: object | None) -> object | None:
    """Recursively remove dict objects where type is set to 'reasoning'."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        if obj.get("type") == "reasoning":
            return None
        return {
            k: v
            for k, v in ((k, _remove_reasoning_objects(v)) for k, v in obj.items())
            if v is not None
        }
    if isinstance(obj, list):
        return [v for v in (_remove_reasoning_objects(i) for i in obj) if v is not None]
    return obj


class HermesAgentLLM(BaseDiagnosticLLM):
    """Override LLM to remove reasoning tags which causes errors with Hermes."""

    @override
    def _preprocess_messages(self, message_dicts: list[dict]) -> list[dict]:
        return _remove_reasoning_objects(message_dicts)

    @override
    def _get_diag_prefix(self) -> str:
        return "HermesAgentLLM"

    @override
    def _get_env_keys(self) -> tuple[str, str]:
        return ("HERMES_AGENT_BASE_URL", "HERMES_AGENT_API_KEY")


class HermesAgentPersona(BaseChatPersona):
    """HermesAgentPersona - Uses BaseChatPersona special coded for HermesAgent."""

    default_model_id: str = "openai/hermes-agent"
    base_url_key: str = "HERMES_AGENT_BASE_URL"
    base_url_default: str = "http://localhost:8649/v1"
    api_key_key: str = "HERMES_AGENT_API_KEY"
    avatar_file: str = "hermes-agent.svg"
    persona_name: str = "HermesAgent"
    llm_class: type = HermesAgentLLM
