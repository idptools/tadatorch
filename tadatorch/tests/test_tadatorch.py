"""
Unit and regression test for the tadatorch package.
"""

import sys

import tadatorch


def test_tadatorch_imported() -> None:
    """Sample test, will always pass so long as import statement worked."""
    assert "tadatorch" in sys.modules


def test_user_facing_functions_are_exposed() -> None:
    """The prediction functions should be importable from the top of the package."""
    assert callable(tadatorch.predict)
    assert callable(tadatorch.predict_from_fasta)
