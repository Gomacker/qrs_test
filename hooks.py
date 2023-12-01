from collections import defaultdict
from typing import Any, Callable


class HookManager:
    def __init__(self):
        self._hooks: dict = defaultdict(list)

    def on(self, name: str):
        def wrapper(func: Callable) -> Callable:
            self._hooks[name].append(func)
            return func

        return wrapper

    def emit(self, name: str, *args, **kwargs) -> None:
        if name in self._hooks:
            for hook_func in self._hooks[name]:
                hook_func(*args, **kwargs)

    def on_face_inference(self):
        return self.on('face_inference')


hook_manager = HookManager()
