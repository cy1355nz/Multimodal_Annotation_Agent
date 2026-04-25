"""
Model factory for creating chat models and embedding models.
"""
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from utils.config_handler import model_conf


class BaseModelFactory(ABC):
    """Abstract base class for model factories."""

    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """Generate model instance."""
        pass


class ChatModelFactory(BaseModelFactory):
    """Factory for creating chat models."""

    def generator(self) -> Optional[BaseChatModel]:
        """Create and return a chat model instance."""
        return ChatTongyi(
            model=model_conf["chat_model_name"],
            temperature=model_conf.get("temperature", 0.1),
            max_tokens=model_conf.get("max_tokens", 2000)
        )


# Global chat model instance
chat_model = ChatModelFactory().generator()
