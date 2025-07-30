#!/usr/bin/env python
"""
Backward compatibility wrapper for server
"""
import sys
import os

# Add src to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and start the server
if __name__ == "__main__":
    from src.ai_classification.api.server import app
    import uvicorn
    print("Starting AI Classification Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)