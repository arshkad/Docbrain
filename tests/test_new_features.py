"""
Tests for: analytics tracking, conversation history (multi-turn RAG),
bulk document operations, and tagging.
Run: pytest tests/test_new_features.py -v
"""

import sys
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
