import time


def get_timestamp() -> str:
    """
    Generates a timestamp string in the format 'YYYY-MM-DD_HH-MM-SS'.

    Returns:
        str: The formatted timestamp string.
    """
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())