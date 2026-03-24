import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import AppRequestHandler

class handler(AppRequestHandler):
    pass
