"""
Visualization Module
====================

Provides interactive UI components for exploring the research analysis.

Components:
- Topic input
- Paper list
- Comparative tables
- Trend plots
- Knowledge graph visualization
"""

# Expose the app path for easy running
import os
APP_PATH = os.path.join(os.path.dirname(__file__), "app.py")

__all__ = ["APP_PATH"]
