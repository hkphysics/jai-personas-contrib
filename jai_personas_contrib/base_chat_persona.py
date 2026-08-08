"""
Base class for Chat Personas
base_chat_persona.py
"""

import os
from pathlib import Path
from typing import override

from jupyter_ai_persona_manager import PersonaDefaults
from jupyter_ai_jupyternaut.jupyternaut.chat_models import ChatLiteLLM
from jupyter_ai_jupyternaut.jupyternaut.jupyternaut import (
    JupyternautPersona,
    DEFAULT_MODEL_ID as JUPYTERNAUT_DEFAULT_MODEL_ID
)
from langgraph.checkpoint.memory import MemorySaver as InMemorySaver
from langchain.agents import create_agent


class BaseChatPersona(JupyternautPersona):
    """
    Base class for chat personas with common functionality
    """

    default_model_id = "placeholder"
    base_url_key = "PLACEHOLDER_BASE_URL"
    base_url_default = "http://localhost:8000/v1"
    api_key_key = "API_KEY"
    avatar_file = "default_avatar.svg"
    persona_name = "BaseChatPersona"
    llm_class = ChatLiteLLM

    @property
    def defaults(self) -> PersonaDefaults:
        """ Create defaults """
        avatar_path = str(
            Path(__file__).parent / "static" / self.avatar_file
        )
        return PersonaDefaults(
            name=self.persona_name,
            avatar_path=avatar_path,
            description=f"An agent that calls {self.persona_name}",
            system_prompt=""
        )

    @override
    def _resolve_model(self) -> tuple[str | None, dict[str, object]]:
        """ override model resolve """
        selected = self.get_model()
        if selected is not None and selected != JUPYTERNAUT_DEFAULT_MODEL_ID:
            params = self._read_config_fields().get(selected, {})
            return selected, dict(params)

        api_base = os.environ.get(
            self.base_url_key,
            self.base_url_default
        )
        api_key = os.environ.get(
            self.api_key_key, ""
        )

        return self.default_model_id, {
            "api_base": api_base,
            "api_key": api_key
        }

    @override
    async def get_agent(self, model_id: str, model_args, system_prompt: str):
        """ Get agent and force no memory """
        model = self.llm_class(**model_args, model=model_id, streaming=True)
        memory_store = InMemorySaver()

        return create_agent(
            model,
            system_prompt=system_prompt,
            checkpointer=memory_store,
            tools=await self.get_tools(),
            middleware=[self._create_tool_error_handler()],
        )
