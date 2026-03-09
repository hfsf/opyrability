import sys
import os

# Ensure src/ and tests/ are on the path so that both opyrability and
# test helper modules (shower, dma_mr, etc.) can be imported directly.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
TESTS = os.path.join(ROOT, "tests")

for p in (SRC, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
