"""Domain exceptions - services raise these; routers map them to HTTP responses.

Every message is written for the end user: it says what went wrong AND what
to do about it.
"""


class BlueprintError(Exception):
    """Base for all domain errors; message is safe to show to the user."""


class UnsupportedFileType(BlueprintError):
    pass


class FileTooLarge(BlueprintError):
    pass


class InvalidFile(BlueprintError):
    """File matched a supported extension but could not be parsed."""


class ExtractionFailed(BlueprintError):
    pass


class VisionUnavailable(BlueprintError):
    """Image extraction requires a vision model that is not reachable."""


class FileNotFound(BlueprintError):
    pass


class AlreadyIngested(BlueprintError):
    pass


class RenderFailed(BlueprintError):
    pass
