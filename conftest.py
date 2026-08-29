import os
import sys

# Ensure the project root is importable when running pytest from any cwd.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
