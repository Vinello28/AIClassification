#!/usr/bin/env python
"""
Backward compatibility wrapper for client
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the refactored client
from src.ai_classification.api.client import *

if __name__ == "__main__":
    # Test the client
    client = AIClassificationClient()
    print("AI Classification Client loaded successfully!")
    if client.is_server_healthy():
        print("Server is healthy!")
    else:
        print("Server is not running. Start it with: python server.py")