"""
LangChain tools for retrieval-related workflow steps.
"""
import json

from langchain_core.tools import tool

from agent.rag import SimpleRAGStore
from utils.logger_handler import logger


@tool(description="Retrieve relevant annotation examples and approved cases from the local RAG store")
def retrieve_annotation_context(query: str, k: int = 3) -> str:
    """
    Retrieve RAG context for an annotation request.

    Args:
        query: Natural language scene description.
        k: Number of context items to return.

    Returns:
        JSON string containing retrieved context items.
    """
    store = SimpleRAGStore.from_project_defaults()
    contexts = store.retrieve(query, k=k)
    payload = [context.model_dump(mode="json") for context in contexts]
    logger.info("[tool] retrieve_annotation_context returned %s contexts", len(payload))
    return json.dumps(payload, ensure_ascii=False)
