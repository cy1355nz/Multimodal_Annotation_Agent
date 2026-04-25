"""
Middleware for annotation agent.
Provides logging, monitoring, and context management.
"""
from utils.logger_handler import logger
from typing import Callable
from langchain.agents.middleware import wrap_tool_call, before_model
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

@wrap_tool_call
def log_tool_calls(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]):
    """
    Middleware to log tool calls for monitoring and debugging.

    Args:
        state: Current agent state.

    Returns:
        Updated state.
    """
    logger.info(f"[tool monitor] Tool used：{request.tool_call['name']}")
    logger.info(f"[tool monitor] Args used：{request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"[tool monitor] Tool {request.tool_call['name']} succeed!")

        return result
    except Exception as e:
        logger.error(f"Tool {request.tool_call['name']} failed, due to：{str(e)}")
        raise e

@before_model
def log_before_model(state: dict, runtime: Runtime):
    """
    Middleware to log before model inference.

    Args:
        state: Current agent state.

    Returns:
        Updated state.
    """
    logger.debug("[Middleware] Before model inference")
    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | "
                 f"{state['messages'][-1]}")
    return None
