import os
import importlib
from typing import List
from colrag.plugins.base import BasePlugin

def load_plugins() -> List[BasePlugin]:
    plugins = []
    plugin_dir = os.path.dirname(__file__)
    for filename in os.listdir(plugin_dir):
        if filename.endswith('.py') and filename != '__init__.py' and filename != 'base.py':
            module_name = f"colrag.plugins.{filename[:-3]}"
            module = importlib.import_module(module_name)
            for item_name in dir(module):
                item = getattr(module, item_name)
                if isinstance(item, type) and issubclass(item, BasePlugin) and item != BasePlugin:
                    plugins.append(item())
    return plugins