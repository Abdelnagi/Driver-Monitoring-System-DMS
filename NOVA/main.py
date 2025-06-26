"""
@file main.py
@brief Initializes the Eel frontend, launches the assistant interface in browser,
       and starts the background monitoring thread.
"""

# === IMPORTS ===
import eel                      # For Python <-> JS communication
import webbrowser               # To open the frontend in default browser
import threading                # For non-blocking background execution

from engine.features import *   # Contains monitoring_loop and related backend logic
from engine.command import *    # Contains assistant command logic (e.g., playAssistantSound)

# === FRONTEND INITIALIZATION ===

# Set Eel's frontend folder (must contain index.html)
eel.init("www")

# Play assistant startup sound
playAssistantSound()

# Open the assistant interface in the default web browser
webbrowser.open_new("http://localhost:8000/index.html")

# Start the real-time monitoring loop in a background thread
threading.Thread(target=monitoring_loop, daemon=True).start()

# Start the Eel server (serves the frontend, enables JS <-> Python functions)
eel.start(
    'index.html',     # Entry HTML file
    mode=None,        # No new browser window (we opened it manually)
    host='localhost',
    port=8000,
    block=True        # Keeps the app running
)
