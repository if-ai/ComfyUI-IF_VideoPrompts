#init.py
import os
import sys

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import necessary modules
import folder_paths

# Import the module with relative import
from . import IF_VideoPromptsNode
VideoPromptNode = IF_VideoPromptsNode.VideoPromptNode

NODE_CLASS_MAPPINGS = {
    "VideoPromptNode": VideoPromptNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoPromptNode": "IF Video Prompts 🎥🧠"
}

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]