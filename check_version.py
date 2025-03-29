# check_version.py
import importlib.metadata
print("openai version:", importlib.metadata.version("openai"))
