from datetime import datetime


def get_experiment_name(name: str) -> str:
    now = datetime.now()
    return f"{name}_{now.strftime('%Y%m%d_%H%M%S')}"
