# Import core functions and classes
from .core import *

# Import version number from pkg_resources
# NOTE: this is apparently slow. Consider parsing from a _version.py instead
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass