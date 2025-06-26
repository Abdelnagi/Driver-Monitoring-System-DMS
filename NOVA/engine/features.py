"""
@file features.py
@brief Provides UI audio feedback and cross-platform automation features, including:
       - Assistant sound effects
       - WhatsApp message/call automation
       - Google Maps integration with Eel frontend binding
"""

# === IMPORTS ===

# Core system
import os
import time
import subprocess
import json
from urllib.parse import quote

# External packages
import pygame                # For playing sound effects
import eel                   # Python <-> JS bridge
import pyautogui             # Automates GUI actions (e.g., clicking, pressing keys)

# Internal modules
from engine.command import AudioManager, state  # Audio feedback + assistant state flags


# === AUDIO FEEDBACK (Frontend Interaction Sounds) ===

@eel.expose
def playAssistantSound():
    """
    Plays a startup sound when the assistant activates.
    Called once at app launch.
    """
    music_path = "www/assets/audio/start_sound.mp3"
    pygame.mixer.init()
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


@eel.expose
def playClickSound():
    """
    Plays a click sound for UI feedback (e.g., when clicking a button).
    """
    music_path = "www/assets/audio/click_sound.wav"
    pygame.mixer.init()
    pygame.mixer.music.load(music_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


# === WHATSAPP AUTOMATION (Windows-specific) ===

def send_whatsApp_msg(mobile_no, message, flag, name):
    """
    Sends a WhatsApp message or initiates a call on Windows.

    Args:
        mobile_no (str): Recipient's phone number in international format.
        message (str): Message content.
        flag (str): 'message' to send text, 'call' for voice, anything else for video call.
        name (str): Contact name (for voice feedback, optional).
    """
    encoded_message = quote(message)
    whatsapp_url = f"whatsapp://send?phone={mobile_no}&text={encoded_message}"
    command = f'start "" "{whatsapp_url}"'

    subprocess.run(command, shell=True)
    time.sleep(6)  # Wait for app to open

    if flag == 'message':
        pyautogui.press('enter')
        time.sleep(1)
        pyautogui.hotkey('alt', 'f4')  # Close window
    elif flag == 'call':
        pyautogui.click(x=1807, y=114)  # Adjust coordinates as needed
        time.sleep(7)
        pyautogui.hotkey('alt', 'f4')
    else:
        pyautogui.click(x=1200, y=100)  # Video call (coordinates may vary)
        time.sleep(1)


# === GOOGLE MAPS HANDLING ===

@eel.expose
def OpenGps(query):
    """
    Opens Google Maps in browser using coordinates from 'location.json'.
    Can also open custom queries (e.g., "restaurant").

    Args:
        query (str): Keyword to trigger map or custom browser behavior.
    """
    audio = AudioManager()

    if query and ("gps" in query or "map" in query or "location" in query):
        audio.speak("Opening your location on Google Maps")

        try:
            with open("location.json", "r") as f:
                data = json.load(f)
                lat = data.get("latitude")
                lon = data.get("longitude")

                if lat and lon:
                    map_url = f"https://www.google.com/maps?q={lat},{lon}"
                    subprocess.Popen(["cmd", "/c", "start", "", map_url])
                else:
                    audio.speak("Location coordinates not available.")
        except Exception as e:
            print(f"[Location Error] {e}")
            audio.speak("Couldn't open Google Maps.")

    elif query:
        audio.speak(f"Opening {query}")
        os.system(f"bash -i -c 'start {query}'")
    else:
        audio.speak("Please tell me what to open")


@eel.expose
def CloseMaps():
    """
    Attempts to close any open Google Maps window by matching its title.
    Windows only — requires 'wmctrl' for Linux if extended.
    """
    audio = AudioManager()
    try:
        os.system("wmctrl -c 'Google Maps'")
        os.system("wmctrl -c 'maps'")
        audio.speak("Closed Google Maps window.")
    except Exception as e:
        print(f"[Error closing maps]: {e}")
        audio.speak("Couldn't close Google Maps.")
