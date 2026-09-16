"""
Openclaw persona
"""

import logging
import os
from typing import override

from jupyter_ai_jupyternaut.jupyternaut.chat_models import ChatLiteLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from .base_chat_persona import BaseChatPersona


class OpenclawLLM(ChatLiteLLM):
    """Override ChatLiteLLM to add diagnostics on failure."""

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
        params = {**params, **kwargs}
        try:
            response = self.completion_with_retry(
                messages=message_dicts, run_manager=run_manager, **params
            )
        except Exception as exc:
            diag = (
                f"OpenclawLLM error: model={getattr(self,'model','?')} "
                f"params={params} env_base={os.environ.get('OPENCLAW_BASE_URL','<unset>')} "
                f"env_key_present={'OPENCLAW_API_KEY' in os.environ} "
                f"original={exc}"
            )
            logging.getLogger(__name__).error(diag, exc_info=True)
            raise type(exc)(f"{diag}\nCaused by: {exc}") from exc
        return self._create_chat_result(response)


class OpenclawPersona(BaseChatPersona):
    """
    Subclass of BaseChatPersona coded to connect to openclaw
    """
    default_model_id = "openai/openclaw"
    base_url_key = "OPENCLAW_BASE_URL"
    base_url_default = "http://localhost:18789/v1"
    api_key_key = "OPENCLAW_API_KEY"
    avatar_file = "openclaw.svg"
    persona_name = "Openclaw"
    llm_class = OpenclawLLM

    @override
    async def get_tools(self) -> list:
        """Filter out the bash tool that causes errors with openclaw."""
        tools = await super().get_tools()
        # Remove the bash tool that causes errors
        return [
            tool
            for tool in tools
            if getattr(tool, "__name__", None) != "bash"
        ]