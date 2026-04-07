"""
Shared pipeline configuration
==============================
Set by main.py at startup; read by downstream scripts.

TARGET_MODE controls which regression target is used:
  "stability" → ln(TS80m)    (default)
  "PCE"       → JV_default_PCE

Each script's main() checks this module to determine the active target.
"""

# ---------------------------------------------------------------------------
# Target mode — set by main.py before importing other modules
# ---------------------------------------------------------------------------
TARGET_MODE = "stability"   # "stability" or "PCE"


def target_column() -> str:
    """Return the column name used as regression target Y."""
    if TARGET_MODE == "PCE":
        return "JV_default_PCE"
    return "lnTS80m"


def target_label() -> str:
    """Return a human-readable label for plots and filenames."""
    if TARGET_MODE == "PCE":
        return "PCE"
    return "lnTS80m"


def target_file_tag() -> str:
    """Return a short tag used in output filenames to distinguish targets."""
    if TARGET_MODE == "PCE":
        return "PCE"
    return "lnTS80m"
