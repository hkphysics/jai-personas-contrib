import os
from jupyter_ai_persona_manager import PersonaDefaults
from jupyter_ai_jupyternaut.jupyternaut.jupyternaut import JupyternautPersona
from jupyter_ai_jupyternaut.jupyternaut.chat_models import ChatLiteLLM
from langgraph.checkpoint.memory import MemorySaver as InMemorySaver                                                                                                                              
from langchain.agents import create_agent

class OpenclawPersona(JupyternautPersona):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def defaults(self) -> PersonaDefaults:
        avatar_path = str(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "static", "openclaw.svg"
                )
            )
        )

        return PersonaDefaults(
            name = "Openclaw",
            avatar_path = avatar_path,
            description = "An agent that calls openclaw",
            system_prompt = ""
        )

    async def get_agent(self, model_id: str, model_args, system_prompt: str):
        memory_store = InMemorySaver() # do not persist queries
        model = ChatLiteLLM(**model_args, model=model_id, streaming=True)

        return create_agent(
            model,
            system_prompt=system_prompt,
            checkpointer=memory_store,
            tools=await self.get_tools(),
            middleware=[self._create_tool_error_handler()]
        )
