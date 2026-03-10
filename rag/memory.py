def append_history(history: str, user_msg: str, assistant_msg: str) -> str:
    return history + f"User: {user_msg}\nAssistant: {assistant_msg}\n"