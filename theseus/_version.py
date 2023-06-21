import warnings

from torch import __version__ as _torch_version
from semantic_version import Version

if Version(_torch_version) < Version("2.0.0"):
    warnings.warn(
        "Using torch < 2.0 is deprecated and support will be discontinued "
        " in future releases.",
        FutureWarning,
    )

__version__ = "0.2.0.dev0"
