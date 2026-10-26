"""Unitree G1 29-DoF + Dex3-1 robot description: URDFs, joint table and DDS layout."""

from .joints import BODY, LEFT_HAND, RIGHT_HAND, Joint  # noqa: F401
from .layout import *  # noqa: F401,F403
from .paths import assets_dir, urdf_path  # noqa: F401
