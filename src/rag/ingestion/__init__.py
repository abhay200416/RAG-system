__all__ = ["SMART_PARSER"]


def __getattr__(name: str):
    if name == "SMART_PARSER":
        from .loader.smart_processor import SMART_PARSER

        return SMART_PARSER
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")