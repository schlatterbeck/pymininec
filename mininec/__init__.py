try:
    from .Version import VERSION as __version__
except ImportError:
    __version__ = '0+unknown'
