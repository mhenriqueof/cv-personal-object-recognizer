"""
Main Entry Point

Launches the personal object recognition system - a system for learning and recognizing objects in
real-time.
"""

from src.app import PersonalObjectRecognizer

app = PersonalObjectRecognizer()
app.run(show_fps=True)
