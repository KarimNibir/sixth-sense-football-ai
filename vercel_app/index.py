"""
Vercel serverless entry point — wraps the FastAPI app.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.server import app
