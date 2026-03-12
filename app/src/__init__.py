import sys
import os

module_path = os.path.abspath('/app/src')

if module_path not in sys.path:
    sys.path.append(module_path)