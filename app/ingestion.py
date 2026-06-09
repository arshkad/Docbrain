"""
Document ingestion pipeline.
Handles parsing (PDF/text), chunking, and storing in the vector database.
"""

import re
import uuid
import hashlib
from pathlib import Path
from typing import BinaryIO

from PyPDF2 import PdfReader

from app.config import settings
from app.database import get_or_create_collection