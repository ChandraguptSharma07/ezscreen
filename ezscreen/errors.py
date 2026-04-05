from __future__ import annotations


class EzscreenError(Exception):
    """Base class for all ezscreen exceptions."""


# ---------------------------------------------------------------------------
# User input errors — caught before any computation, no quota spent
# ---------------------------------------------------------------------------

class UserInputError(EzscreenError):
    """Invalid user-provided input."""


class InvalidReceptorError(UserInputError):
    """PDB file or RCSB code is invalid or cannot be fetched."""


class InvalidLigandInputError(UserInputError):
    """Ligand file or folder is invalid or unreadable."""


class InvalidBoxError(UserInputError):
    """Binding site box definition is invalid (zero volume, NaN coords, etc.)."""


# ---------------------------------------------------------------------------
# Prep errors
# ---------------------------------------------------------------------------

class PrepError(EzscreenError):
    """Base class for preparation pipeline errors."""


class ReceptorPrepError(PrepError):
    """Fatal receptor prep failure. The run cannot continue."""


class LigandPrepError(PrepError):
    """Non-fatal per-ligand prep failure. Logged to failed.sdf and skipped."""


# ---------------------------------------------------------------------------
# Authentication errors
# ---------------------------------------------------------------------------

class AuthError(EzscreenError):
    """Base class for credential and authentication errors."""


class KaggleAuthError(AuthError):
    """Kaggle credential is missing, invalid, or has insufficient permissions."""


class NIMAuthError(AuthError):
    """NVIDIA NIM API key is missing or rejected."""


class CredentialPermissionError(AuthError):
    """Credentials file has insecure permissions (not 600)."""


# ---------------------------------------------------------------------------
# Kaggle API errors — permanent, never retry
# ---------------------------------------------------------------------------

class KaggleError(EzscreenError):
    """Base class for Kaggle API errors."""


class KaggleBadRequestError(KaggleError):
    """HTTP 400 — invalid request parameters."""


class KaggleUnauthorizedError(KaggleError):
    """HTTP 401 — API key rejected or expired."""


class KaggleForbiddenError(KaggleError):
    """HTTP 403 — account needs phone verification."""


class KaggleNotFoundError(KaggleError):
    """HTTP 404 — dataset or kernel does not exist."""


class KaggleQuotaExhaustedError(KaggleError):
    """Weekly GPU quota exhausted."""


class KaggleStorageLimitError(KaggleError):
    """Kaggle dataset storage limit reached."""


# ---------------------------------------------------------------------------
# Kaggle API errors — transient, retry with backoff
# ---------------------------------------------------------------------------

class KaggleRateLimitError(KaggleError):
    """HTTP 429 — rate limited. Retry with exponential backoff + Retry-After."""


class KaggleServerError(KaggleError):
    """HTTP 5xx — transient server error. Retry with exponential backoff, max 3."""


class KernelPreemptedError(KaggleError):
    """Kaggle kernel preempted mid-run. Retry up to shard_retry_limit."""


# ---------------------------------------------------------------------------
# Network errors — transient
# ---------------------------------------------------------------------------

class NetworkTimeoutError(EzscreenError):
    """Network request timed out. Retry once after 3 seconds."""


# ---------------------------------------------------------------------------
# Internal errors — fatal, always show log path and GitHub issues URL
# ---------------------------------------------------------------------------

class InternalError(EzscreenError):
    """Unexpected internal error."""


class CheckpointError(InternalError):
    """SQLite checkpoint read or write failure."""


class TemplateRenderError(InternalError):
    """Jinja2 notebook template rendering failure."""
