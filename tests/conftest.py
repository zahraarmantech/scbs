"""Pytest configuration — adds src/ to path so tests can import scbs."""

import sys
import os

_HERE = os.path.dirname(__file__)
_SRC  = os.path.join(_HERE, "..", "src")
sys.path.insert(0, os.path.abspath(_SRC))
