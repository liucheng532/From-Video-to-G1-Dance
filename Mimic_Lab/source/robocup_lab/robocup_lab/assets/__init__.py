import os

try:
    from .ball import SOCCER_BALL_CFG  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - allows lightweight imports without Isaac Lab
    SOCCER_BALL_CFG = None

# Conveniences to other module directories via relative paths
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))

__all__ = ["ASSET_DIR", "SOCCER_BALL_CFG"]
