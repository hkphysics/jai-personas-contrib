"""
Base class for Chat Personas
base_chat_persona.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import override

from jupyter_ai_jupyternaut.jupyternaut.chat_models import ChatLiteLLM
from jupyter_ai_jupyternaut.jupyternaut.jupyternaut import (
    DEFAULT_MODEL_ID as JUPYTERNAUT_DEFAULT_MODEL_ID,
)
from jupyter_ai_jupyternaut.jupyternaut.jupyternaut import JupyternautPersona
from jupyter_ai_persona_manager import PersonaDefaults
from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langgraph.checkpoint.memory import MemorySaver as _InMemorySaver

_MEMORY_SAVER = _InMemorySaver()


class BaseDiagnosticLLM(ChatLiteLLM):
    """
    Base LLM class with diagnostic error handling.
    Subclasses can override _preprocess_messages, base_url_key, and api_key_key.
    """

    def _preprocess_messages(self, message_dicts: list[dict]) -> list[dict]:
        """Hook to preprocess message dicts before completion."""
        return message_dicts

    base_url_key: str = ""
    api_key_key: str = ""

    @override
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
            stream_iter = self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        message_dicts = self._preprocess_messages(message_dicts)
        params = {**params, **kwargs}
        try:
            response = self.completion_with_retry(
                messages=message_dicts, run_manager=run_manager, **params
            )
        except Exception as exc:
            env_base = (
                os.environ.get(self.base_url_key, "<unset>") if self.base_url_key else "<unset>"
            )
            env_key_present = self.api_key_key in os.environ if self.api_key_key else False
            diag = (
                f"{self.__class__.__name__} error: "
                f"model={getattr(self, 'model', '?')} "
                f"params={params} env_base={env_base} "
                f"env_key_present={env_key_present} "
                f"original={exc}"
            )
            logging.getLogger(__name__).exception(diag)
            raise type(exc)(f"{diag}\nCaused by: {exc}") from exc
        return self._create_chat_result(response)


class BaseChatPersona(JupyternautPersona):
    """
    Base class for chat personas with common functionality
    """

    default_model_id: str = "placeholder"
    base_url_key: str = "PLACEHOLDER_BASE_URL"
    base_url_default: str = "http://localhost:8000/v1"
    api_key_key: str = "API_KEY"
    avatar_file: str = "default_avatar.svg"
    persona_name: str = "BaseChatPersona"
    llm_class: type[ChatLiteLLM] = ChatLiteLLM

    @property
    def defaults(self) -> PersonaDefaults:
        """Return default persona metadata."""
        avatar_path = str(Path(__file__).parent / "static" / self.avatar_file)
        return PersonaDefaults(
            name=self.persona_name,
            avatar_path=avatar_path,
            description=f"An agent that calls {self.persona_name}",
            system_prompt="",
        )

    @override
    def _resolve_model(self) -> tuple[str | None, dict[str, object]]:
        """override model resolve"""
        selected = self.get_model()
        if selected is not None and selected != JUPYTERNAUT_DEFAULT_MODEL_ID:
            params = self._read_config_fields().get(selected, {})
            return selected, dict(params)

        api_base = os.environ.get(self.base_url_key, self.base_url_default)
        api_key = os.environ.get(self.api_key_key, "")

        logging.getLogger(__name__).debug(
            "DIAG: BaseChatPersona resolving model: selected=%s api_base=%s api_key_present=%s",
            selected,
            api_base,
            self.api_key_key in os.environ,
        )

        return self.default_model_id, {"api_base": api_base, "api_key": api_key}

    @override
    async def get_agent(
        self, model_id: str, model_args: dict[str, object], system_prompt: str
    ) -> object:
        """Build the agent with a fresh in-memory checkpointer."""
        model = self.llm_class(**model_args, model=model_id, streaming=True)

        return create_agent(
            model,
            system_prompt=system_prompt,
            checkpointer=_MEMORY_SAVER,
            tools=await self.get_tools(),
            middleware=[self._create_tool_error_handler()],
        )
