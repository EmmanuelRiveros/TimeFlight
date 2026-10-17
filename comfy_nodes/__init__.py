"""ComfyUI-TimeFlight node pack: thin wrappers over timeflight/.

Install: link this folder into ComfyUI/custom_nodes/ and `pip install -e` the TimeFlight repo
(colab/setup_comfy.sh does both).
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
