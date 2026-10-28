import os
def is_deepspeed_enabled():
    return False
def get_env_args(name, type_func, default=None):
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return type_func(v)
    except Exception:
        return default
