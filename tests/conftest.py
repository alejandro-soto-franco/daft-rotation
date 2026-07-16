from __future__ import annotations

import pytest
from daft.session import Session

import daft_rotation


@pytest.fixture
def sess():
    s = Session()
    s.load_extension(daft_rotation)
    with s:
        yield s
