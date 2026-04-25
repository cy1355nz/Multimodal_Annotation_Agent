import os


def get_project_root() -> str:
    """
    Get the root directory of the project.

    Returns:
        String path of the project root directory.
    """
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(current_dir)

    return project_root


def get_abs_path(relative_path: str) -> str:
    """
    Get absolute path from relative path.

    Args:
        relative_path: Relative path string.

    Returns:
        Absolute path string.
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    print(get_abs_path("config/model.yml"))