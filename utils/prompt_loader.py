"""
Prompt loader utility for loading system prompts from files.
"""
from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path


def load_main_prompt() -> str:
    """Load main system prompt from file."""
    prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    return _read_prompt_file(prompt_path)


def load_few_shot_examples() -> str:
    """Load few-shot examples from file."""
    few_shot_path = get_abs_path(prompts_conf["few_shot_path"])
    return _read_prompt_file(few_shot_path)


def _read_prompt_file(file_path: str) -> str:
    """
    Read prompt file content.

    Args:
        file_path: Path to the prompt file.

    Returns:
        Content of the prompt file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise FileNotFoundError(f"Failed to load prompt file {file_path}: {str(e)}")


def load_system_prompts() -> str:
    """
    Load and combine all system prompts.

    Returns:
        Combined system prompt string.
    """
    main_prompt = load_main_prompt()
    few_shot = load_few_shot_examples()

    return f"{main_prompt}\n\n{few_shot}"
