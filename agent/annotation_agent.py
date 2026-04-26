"""
Multimodal Annotation Agent core implementation.
Handles annotation workflow with tool calling, RAG, and structured output.
"""
import json
from typing import List, Optional, Dict, Any
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from models.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.annotation_tools import (
    read_text_file,
    analyze_image,
    query_vehicle_data,
    validate_json_output,
    save_annotation_result
)
from agent.tools.middleware import log_tool_calls, log_before_model
from schemas.annotation_schema import AnnotationResult


class AnnotationAgent:
    """
    Multimodal annotation agent for autonomous driving scene labeling.

    This agent converts natural language descriptions from annotators
    into structured JSON annotations using VLM and tool calling.
    """

    def __init__(self):
        """Initialize the annotation agent with model, prompts, and tools."""

        self.system_prompt_content = load_system_prompts()
        self.agent = create_agent(
            model=chat_model,
            tools=[
                # read_text_file,
                analyze_image,
                query_vehicle_data,
                validate_json_output,
                save_annotation_result
            ],
            middleware=[log_tool_calls, log_before_model],
        )

    def _prepare_messages(self, description: str, image_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Prepare messages with injected paths to prevent LLM hallucination.
        This ensures the LLM sees the exact paths in the text context.
        """
        messages = []

        prompt_used = f"\n**Input Description:**\n{description}\n"

        # debug only
        # image_paths = ['/Users/leo/Desktop/Resume/Resume_Prep/Agent-Study/Demo/Multimodal_Annotation_Agent/data/temp/n008-2018-03-14-15-16-29-0400__CAM_FRONT__1521055565282267.jpg']
        if image_paths:
            # 1. Inject paths as text for the Agent's tool calling logic
            path_text = "\n**Available Image Paths for Tools**\n"
            for idx, path in enumerate(image_paths):
                path_text += f"- image_{idx + 1}_path: {path}\n"

            path_text += ("\n**Instruction for Using Images**: When calling 'analyze_image', you MUST use the exact "
                          "'image_X_path' values provided above.")

            prompt_used += path_text
            content_parts = [{"type": "text", "text": prompt_used}]

            # 2. Inject actual image blocks for VLM vision
            for path in image_paths:
                content_parts.append({
                    "type": "image",
                    "image": path
                })
        else:
            content_parts = [{"type": "text", "text": prompt_used}]

        messages.append(SystemMessage(content=self.system_prompt_content))
        messages.append(HumanMessage(content=content_parts))
        return messages

    def execute_stream(self, description: str, image_paths: list = None):
        """
        Execute annotation workflow with streaming output.
        """
        # prepare messages with injected paths
        messages = self._prepare_messages(description, image_paths)
        input_dict = {"messages": messages}

        # Stream agent responses
        # omit tool messages
        for chunk in self.agent.stream(input_dict, stream_mode="values"):
            latest_message = chunk["messages"][-1]
            if latest_message.type != 'tool' and latest_message.content:
                yield f"\n\n**--- {latest_message.type.upper()} Message ---**\n\n"
                if isinstance(latest_message.content[0], dict):
                    yield latest_message.content[0]['text'].strip()
                elif isinstance(latest_message.content[0], str):
                    yield latest_message.content[0].strip()
                else:
                    yield latest_message.content

    # def execute_with_structured_output(self, description: str, image_paths: list = None,
    #                                    output_path: str = None) -> AnnotationResult:
    #     """
    #     Execute annotation and return structured Pydantic object.
    #
    #     Args:
    #         description: Natural language description of the scene.
    #         image_paths: Optional list of image paths.
    #         output_path: Optional path to save JSON output.
    #
    #     Returns:
    #         AnnotationResult Pydantic object.
    #     """
    #     # Build input content
    #     content = [{"type": "text", "text": description}]
    #
    #     if image_paths:
    #         for img_path in image_paths[:3]:
    #             content.append({
    #                 "type": "image_url",
    #                 "image_url": {"url": img_path}
    #             })
    #
    #     input_dict = {
    #         "messages": [
    #             {"role": "user", "content": content}
    #         ]
    #     }
    #
    #     # Collect full response
    #     full_response = ""
    #     for chunk in self.chain.stream(input_dict, stream_mode="values"):
    #         latest_message = chunk["messages"][-1]
    #         if latest_message.content:
    #             full_response += latest_message.content
    #
    #     # Extract JSON from response
    #     json_str = self._extract_json(full_response)
    #
    #     # Validate and parse
    #     try:
    #         data = json.loads(json_str)
    #         result = AnnotationResult.model_validate(data)
    #
    #         # Save if output path provided
    #         if output_path:
    #             with open(output_path, 'w', encoding='utf-8') as f:
    #                 json.dump(data, f, ensure_ascii=False, indent=2)
    #
    #         return result
    #     except Exception as e:
    #         raise ValueError(f"Failed to parse annotation result: {str(e)}")
    #
    # def _extract_json(self, text: str) -> str:
    #     """
    #     Extract JSON string from text response.
    #
    #     Args:
    #         text: Full text response from agent.
    #
    #     Returns:
    #         Extracted JSON string.
    #     """
    #     # Try to find JSON between curly braces
    #     start = text.find('{')
    #     end = text.rfind('}')
    #
    #     if start != -1 and end != -1:
    #         return text[start:end + 1]
    #
    #     # If no braces found, return entire text
    #     return text


if __name__ == '__main__':
    agent = AnnotationAgent()

    # Example usage
    description = "You are driving on an urban road and approaching an intersection with traffic lights. The traffic light is currently red, so you need to slow down and come to a complete stop."

    for chunk in agent.execute_stream(description):
        print(chunk, end="", flush=True)
