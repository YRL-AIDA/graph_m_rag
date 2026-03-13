import sys
import os


module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if module_path not in sys.path:
    sys.path.append(module_path)

from app.src.api import run_api

if __name__ == "__main__":
    run_api()
