"""Shared pytest fixtures: paths to made-up tables and an offline settings object."""

from __future__ import annotations

from pathlib import Path

import pytest

from kosis_chat.config import load_settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def mock_settings():
    return load_settings(backend="mock")
