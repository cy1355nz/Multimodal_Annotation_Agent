import yaml
from utils.path_tool import get_abs_path


def load_model_config(config_path: str = get_abs_path("config/model.yml"), encoding: str = "utf-8"):
    """Load model configuration from YAML file."""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8"):
    """Load agent configuration from YAML file."""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding: str = "utf-8"):
    """Load prompts configuration from YAML file."""
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


model_conf = load_model_config()
agent_conf = load_agent_config()
prompts_conf = load_prompts_config()


if __name__ == '__main__':
    print(model_conf["chat_model_name"])