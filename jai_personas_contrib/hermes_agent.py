"""
hermes_agent - Persona for Hermes Agent
"""

from typing import override
from jupyter_ai_jupyternaut.jupyternaut.chat_models import ChatLiteLLM
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult
from langchain_core.language_models.chat_models import generate_from_stream
from .base_chat_persona import BaseChatPersona

def _remove_reasoning_objects(obj: object | None):
    """
    Recursively remove any dictionary objects where type is set to "reasoning".

    Args:
        obj: The object to process (dict, list, or other value)

    Returns:
        The cleaned object with reasoning dicts removed
    """
    if isinstance(obj, dict):
        # Check if this dict has type="reasoning" - remove it entirely
        if obj.get('type') == 'reasoning':
            return None
        return {
            k: v for k, v in (
                (k, _remove_reasoning_objects(val)) for k, val in obj.items()
            )
            if v is not None
        }

    if isinstance(obj, list):
        return [
            v for i in obj if (
                v := _remove_reasoning_objects(i)
            ) is not None
        ]

    # Return primitive values unchanged
    return obj


class HermesAgentLLM(ChatLiteLLM):
    """
    Override ChatLiteLLM to remove reasoning tags which causes errors
    with Hermes
    """
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,
        **kwargs: object,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        message_dicts = _remove_reasoning_objects(message_dicts)
        params = {**params, **kwargs}
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response)


class HermesAgentPersona(BaseChatPersona):
    """
    HermesAgentPersona - Uses BaseChatPersona special coded for
    HermesAgent
    """
    default_model_id = "openai/hermes-agent"
    base_url_key = "HERMES_AGENT_BASE_URL"
    base_url_default = "http://localhost:8649/v1"
    api_key_key = "HERMES_AGENT_API_KEY"
    avatar_file = "hermes-agent.svg"
    persona_name = "HermesAgent"
    llm_class = HermesAgentLLM
