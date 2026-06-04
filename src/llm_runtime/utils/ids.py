from uuid import uuid4


def new_request_id() -> str:
    return f"chatcmpl-{uuid4().hex}"

