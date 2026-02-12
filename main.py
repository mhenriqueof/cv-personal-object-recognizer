"""
Main Entry Point
"""

from src.app import PersonalObjectRecognizer

app = PersonalObjectRecognizer()
app.run(show_fps=True)
