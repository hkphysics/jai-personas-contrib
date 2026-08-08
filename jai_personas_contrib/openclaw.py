"""
Openclaw persona
"""

from typing import override
from .base_chat_persona import BaseChatPersona

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

    @override
    async def get_tools(self):
        """Filter out the bash tool that causes errors with openclaw."""
        tools = await super().get_tools()
        # Remove the bash tool that causes errors
        return [tool for tool in tools if getattr(tool, '__name__', "") != 'bash']
