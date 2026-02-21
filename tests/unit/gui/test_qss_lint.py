"""Lint tests for QSS stylesheets.

Catches duplicate selectors and unsupported CSS properties that slip
through code review.  Runs automatically in CI via pytest.
"""

import re
from pathlib import Path

import pytest

# Directory containing all QSS files
_STYLES_DIR = (
    Path(__file__).resolve().parents[3] / "xpcsviewer" / "gui" / "theme" / "styles"
)

# Collect all QSS files for parametrization
_QSS_FILES = sorted(_STYLES_DIR.glob("*.qss"))


def _extract_selectors(qss_text: str) -> list[tuple[str, int]]:
    """Extract (selector, line_number) pairs from QSS text.

    Returns normalised selectors (stripped, single-spaced) with their
    first occurrence line number.
    """
    selectors: list[tuple[str, int]] = []
    in_comment = False
    accumulator: list[str] = []
    line_start = 0

    for i, line in enumerate(qss_text.splitlines(), start=1):
        # Handle block comments
        while True:
            if in_comment:
                end = line.find("*/")
                if end == -1:
                    line = ""
                    break
                line = line[end + 2 :]
                in_comment = False
            else:
                start = line.find("/*")
                if start == -1:
                    break
                end = line.find("*/", start + 2)
                if end == -1:
                    line = line[:start]
                    in_comment = True
                    break
                line = line[:start] + line[end + 2 :]

        stripped = line.strip()
        if not stripped:
            continue

        # If line contains '{', everything before it is part of the selector
        if "{" in stripped:
            before_brace = stripped.split("{")[0].strip()
            if before_brace:
                accumulator.append(before_brace)
            if accumulator:
                if line_start == 0:
                    line_start = i
                selector = " ".join(accumulator)
                # Normalise whitespace
                selector = re.sub(r"\s+", " ", selector).strip()
                if selector:
                    selectors.append((selector, line_start))
            accumulator = []
            line_start = 0
        elif "}" not in stripped:
            # Continuation of a multi-line selector
            if not accumulator:
                line_start = i
            accumulator.append(stripped)
        else:
            # Closing brace line — reset
            accumulator = []
            line_start = 0

    return selectors


@pytest.mark.parametrize(
    "qss_path",
    _QSS_FILES,
    ids=[p.name for p in _QSS_FILES],
)
class TestQssDuplicateSelectors:
    """Detect duplicate selectors within a single QSS file."""

    def test_no_duplicate_selectors(self, qss_path: Path):
        """Each selector should appear at most once per file."""
        text = qss_path.read_text(encoding="utf-8")
        selectors = _extract_selectors(text)

        seen: dict[str, int] = {}
        duplicates: list[str] = []

        for selector, line_no in selectors:
            if selector in seen:
                duplicates.append(
                    f"  '{selector}' at line {line_no} "
                    f"(first seen at line {seen[selector]})"
                )
            else:
                seen[selector] = line_no

        assert not duplicates, f"Duplicate selectors in {qss_path.name}:\n" + "\n".join(
            duplicates
        )


_UNSUPPORTED_PROPERTIES = re.compile(
    r"^\s*(outline(?:-offset)?|letter-spacing)\s*:", re.MULTILINE
)


@pytest.mark.parametrize(
    "qss_path",
    _QSS_FILES,
    ids=[p.name for p in _QSS_FILES],
)
class TestQssUnsupportedProperties:
    """Detect CSS properties not supported by Qt's QSS engine."""

    def test_no_unsupported_properties(self, qss_path: Path):
        """QSS files should not use outline, outline-offset, or letter-spacing."""
        text = qss_path.read_text(encoding="utf-8")

        # Strip comments first
        text_no_comments = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

        matches = _UNSUPPORTED_PROPERTIES.findall(text_no_comments)
        assert not matches, (
            f"Unsupported CSS properties in {qss_path.name}: "
            f"{sorted(set(matches))}. Qt QSS does not support these."
        )
