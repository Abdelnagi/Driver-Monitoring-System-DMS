# ======================================================
# @file command.py
# @brief Main logic module for the Driver Assistant and Feedback monitoring system.
#        Handles LLM communication, Monitoring feedback of models, voice processing, alerts,
#        email dispatching, location handling, and UI interaction.
# ======================================================


# === CUSTOM GLOBAL EXCEPTION HANDLER ===

import sys
import traceback

def my_excepthook(exctype, value, tb):
    """
    Custom exception hook to print full stack trace
    and wait for user to press Enter before exiting.
    Useful for debugging in console environments.
    """
    traceback.print_exception(exctype, value, tb)
    input("Press Enter to exit...")

# Override default exception handler
sys.excepthook = my_excepthook


# === STANDARD LIBRARY IMPORTS ===

import os                          # File and OS-level utilities
import sys                         # System operations
import traceback                   # Stack trace printing
import time                        # Delay, timestamp
import json                        # File I/O for configuration and session state
import asyncio                     # Async tasks for non-blocking behavior
from datetime import datetime      # Current date/time handling
import re                          # Regex (used for LLM message parsing)
import webbrowser                  # Open links in browser


# === NUMERICAL AND AUDIO PROCESSING ===

import numpy as np                             # Array/matrix math
from pydub import AudioSegment                 # Audio manipulation (merge, export, etc.)
import pygame                                  # Audio playback (clicks, TTS output)


# === LOCATION / GEO SERVICES ===

import geocoder                                # IP-based geolocation
from geopy.geocoders import Nominatim          # Reverse geocoding (lat/lon → address)


# === NETWORKING AND HTTP ===

import requests                                # HTTP requests for APIs (weather, etc.)


# === FRONTEND COMMUNICATION ===

import eel                                     # Python ↔ JS bridge for UI control


# === VOICE ASSISTANT MODELS AND TTS ===

import ollama                                  # Local LLM backend via Ollama (e.g., LLaMA)
import edge_tts                                # Offline Microsoft Edge-based TTS


# === CUSTOM MODULES ===

from send_email import *                      # Email alert system
from config import *                           #  constants, global settings
from groq import Groq                          # Groq API wrapper (optional LLM interface)




# =================== GLOBAL VAR ===================

counter_hands =0

# =========================
# APP STATE CLASS
# =========================

class AppState:
    """
    AppState maintains the global runtime status of the Driver Assistant system.

    === Purpose ===
    This class acts as a shared context container that tracks:
    - Current system mode (e.g., "monitoring" or "assistance")
    - Microphone interaction state
    - LLM conversation history
    - Override GPS coordinates sent from the frontend
    - Runtime flags for JSON monitoring and speech output

    === Attributes ===
    - current_mode (str): "monitoring" or "assistance"
    - mic_pressed (bool): Tracks if the mic button was pressed by the user
    - conversation_history (list): Stores dialogue context for the LLM
    - location_override (tuple or None): Holds GPS coordinates from the browser
    - json_file_path (str): Path to the live monitoring JSON file
    - json_flag (bool): Enables/disables reading the JSON (used to stop monitoring)
    - speak_flag (bool): Enables/disables speech output
    """

    def __init__(self):
        self.current_mode = "monitoring"              # Can be "monitoring" or "assistance"
        self.mic_pressed = False                      # Whether the mic button is currently active
        self.conversation_history = []                # Stores LLM interactions (context)
        self.location_override = None                 # Optional override from frontend GPS
        self.json_file_path = "data\\driver_assistant.json"  # Path to DMS monitoring data file
        self.json_flag = True                         # Flag to enable/disable DMS checks
        self.speak_flag = True                        # Flag to allow/disallow assistant to speak



# =========================
# AUDIO MANAGER CLASS
# =========================

class AudioManager:
    """
    AudioManager handles all sound-related functionality in the assistant system.

    === Purpose ===
    This class is responsible for:
    - Playing predefined audio files (e.g., button clicks, startup sounds)
    - Handling real-time text-to-speech (TTS) using the Microsoft Edge TTS engine (`edge-tts`)
    - Triggering emergency buzzer alerts
    - Controlling playback (pause, resume, stop, volume)

    === Libraries Used ===
    - pygame: for local audio playback (WAV/MP3)
    - edge_tts: for generating speech from text using offline Microsoft voices
    - asyncio: to handle the asynchronous TTS generation
    - os, time: for file path handling and timestamped temp files
    - eel: to expose Python functions to the frontend JavaScript

    === Notes ===
    The `speak()` function is exposed via Eel to allow JavaScript (frontend) to trigger TTS.
    The class uses a global `state` object to stay context-aware of the app’s mode.
    
    """

    def __init__(self):
        self.state = state
        pygame.mixer.init()  # Required before loading/playing audio

    def play(self, path):
        """Play any audio file (WAV or MP3)."""
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

    def stop(self):
        """Immediately stop playback."""
        pygame.mixer.music.stop()

    def pause(self):
        """Pause playback."""
        pygame.mixer.music.pause()

    def unpause(self):
        """Resume paused playback."""
        pygame.mixer.music.unpause()

    def set_volume(self, volume):
        """
        Set the playback volume (float: 0.0 → 1.0).
        """
        pygame.mixer.music.set_volume(volume)

    @eel.expose
    def speak(self, text):
        """
        Public Eel-exposed method to trigger TTS from the frontend.
        Uses edge-tts to speak a message.
        """
        print(f"[TTS] Speaking: {text}")
        asyncio.run(self.edge_speak(text))

    async def edge_speak(self, text, voice="en-US-AriaNeural"):
        """
        Converts the provided text to speech using edge-tts, saves it to a file,
        and plays it using pygame. Uses a temporary folder to store audio files.

        Args:
            text (str): Text to be spoken
            voice (str): Voice model to use (default is 'en-US-AriaNeural')
        """
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            mp3_bytes = b""

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_bytes += chunk["data"]

            # Save the generated MP3 to a temp file
            temp_dir = os.path.join(os.getcwd(), "temp_audio")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, f"temp_audio_{int(time.time())}.mp3")

            with open(temp_file, "wb") as f:
                f.write(mp3_bytes)

            # Play the generated TTS file
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            pygame.mixer.music.stop()
            pygame.mixer.music.unload()

        except Exception as e:
            print(f"[ERROR] Text-to-speech failed: {e}")
            # Fallback: buzzer if TTS fails
            try:
                pygame.mixer.music.load("www/assets/audio/buzzer.wav")
                pygame.mixer.music.play()
            except:
                pass

    def BuzzerSound(self):
        """
        Plays a buzzer alert sound.
        Used during emergencies or fatigue detection.
        """
        music_path = "www/assets/audio/buzzer.wav"
        self.play(music_path)


# =========================
# LLM MANAGER CLASS
# =========================      
class LLMManager():
    """
    LLMManager handles natural language understanding, query processing, and voice interaction
    between the driver and the assistant. It communicates with a local LLM (e.g., LLaMA via Ollama)
    to provide intelligent responses, and routes commands like WhatsApp messaging, navigation, and weather queries.

    === Purpose ===
    - Serve as the central conversation engine.
    - Manage voice input/output (STT/TTS) for driver interaction.
    - Interpret commands and route them to appropriate subsystems (e.g., maps, weather).
    - Track assistant state and update global flags accordingly.

    === Responsibilities ===
    - Initiate and manage LLM context (conversation history, system prompt).
    - Process user speech input and classify intent via prompt engineering.
    - Handle high-level actions: send WhatsApp, open/close GPS, route planning.
    - Call external APIs (Google Maps, OpenWeather) and summarize via LLM.
    - Fall back to small talk (chat intent) if no specific task is detected.
    - Monitor silence to exit assistance mode automatically.

    === Core Methods ===
    - `generate_initial_context()`: Initializes system prompt with time and location.
    - `PassToLlm()`: Main driver-assistant interaction loop.
    - `classify_user_intent()`: Extracts action type and destination from query.
    - `handle_navigation()`, `get_route_info()`: Google Maps integration for routing.
    - `get_weather()`: Weather info via OpenWeatherMap.
    - `geocode_destination()`: Converts place name to coordinates using Maps API.
    - `estimate_tokens()` + `trim_history()`: Keeps message history within token limits.

    === Libraries Used ===
    - `ollama`: To interface with the locally hosted LLaMA model
    - `eel`: For Python <-> JS communication
    - `requests`: For external APIs (Google Maps, OpenWeather)
    - `json`, `time`, `datetime`, `re`: For parsing, formatting, and intent routing
    - `webbrowser`: To open map URLs
    - `geocoder`: For fallback IP geolocation

    === Notes ===
    - This class assumes that `AudioManager` and `User` (speech recognizer) are injected.
    - Heavy use of prompt templates, functional branching, and asynchronous cleanup is performed.

    === Related Components ===
    - `AudioManager`: For speaking responses
    - `User`: STT interface to capture user commands
    - `AppState`: Used for switching between monitoring and assistance modes
    
    """

    def __init__(self, state, Audio, User):
        """
        Initializes the LLMManager with shared application state, 
        audio output handler, and speech-to-text input module.

        Args:
            state (AppState): Global assistant state object (mode, flags, context).
            Audio (AudioManager): Handles text-to-speech and playback feedback.
            User (object): Handles user speech input and command capture (STT).

        On init:
            - Sets the current system prompt in the conversation history.
            - Preloads HERE API key for geolocation services.
        """
        self.state = state
        self.Audio = Audio
        self.User = User

        # Seed the LLM with a system prompt that defines assistant behavior
        self.state.conversation_history = [self.generate_initial_context()]

        # Key for HERE API (used in later geocoding)
        #self.key = HERE_KEY

        
    def generate_initial_context(self):
        """
        Builds the initial system message used to instruct the LLM 
        on its role, current time/date, and user's approximate location.

        Returns:
            dict: Message in format {"role": "system", "content": system_prompt}
        """
        now = datetime.now()
        current_date = now.strftime("%A, %B %d, %Y")
        current_time = now.strftime("%I:%M %p")

        try:
            # Attempt to fetch approximate location via IP lookup
            g = geocoder.ip('me')
            city = g.city or "an unknown city"
            country = g.country or "an unknown country"
            location_text = f"You are currently in {city}, {country}."

            # Override with GPS from frontend if available
            if self.state.location_override:
                location_text = f"You are currently at {self.state.location_override}."
        except Exception as e:
            location_text = "Location is unavailable."
            print(f"[ERROR] Failed to get location: {e}")

        # Compose the system instruction for LLM initialization
        system_prompt = f"""You are a helpful driving assistant.
    Today is {current_date} and the time is {current_time}.
    {location_text}
    You do functions like providing him current location, current time, traffic and weather info, estimated arrival time to destinations and routing steps to these destinations.
    You also talk with him if he feels sleepy or suffers fatigue.
    Only respond in 1–2 sentences unless instructed otherwise.
    """

        return {"role": "system", "content": system_prompt}


    def estimate_tokens(self, text):
        """
        Roughly estimates the number of tokens in a given string.
        
        LLMs like GPT or LLaMA tokenize input based on subwords, 
        but for a fast approximation we assume ~1.3 tokens per word.

        Args:
            text (str): The input string to evaluate.

        Returns:
            float: Estimated number of tokens.
        """
        return len(text.split()) * 1.3


    def trim_history(self, history, max_tokens=7000):
        """
        Trims the conversation history to keep it under a defined token limit.

        This is important for local LLMs (like LLaMA) that have a fixed context window.
        The function retains:
        - The system prompt (first entry)
        - The most recent messages in reverse chronological order until the token cap is hit

        Args:
            history (list): Full conversation history (list of {"role", "content"} dicts).
            max_tokens (int): Maximum allowed token budget for the LLM.

        Returns:
            list: Trimmed history that fits within token constraints.
        """
        total_tokens = 0
        trimmed = []

        # Always keep the system prompt at the beginning
        if history and history[0]['role'] == 'system':
            trimmed.append(history[0])
            history = history[1:]  # Exclude prompt from token count logic

        # Add most recent messages from the end of the history list
        for msg in reversed(history):
            tokens = self.estimate_tokens(msg['content'])
            if total_tokens + tokens <= max_tokens:
                trimmed.insert(1, msg)  # Insert after system message
                total_tokens += tokens
            else:
                break  # Stop when the limit is reached

        return trimmed


    def classify_user_intent(self, query):
        """
        Uses an LLM to classify the user's query into a structured command format.

        Purpose:
            - Extract intent (chat, weather, traffic, ETA, navigation)
            - Extract destination name if applicable
            - This enables clean downstream routing and execution

        Args:
            query (str): Raw user speech or text input.

        Returns:
            dict: JSON object with structure:
                {
                    "type": "chat" | "weather" | "traffic" | "eta" | "navigate",
                    "destination": "Giza" | "New Cairo" | ""
                }

        Notes:
            - Uses prompt-based classification through LLaMA (via Ollama).
            - The model is instructed to respond *only* with valid JSON (no natural language).
            - If classification fails, fallback type is "chat".

        Example output:
            {"type": "navigate", "destination": "maadi"}
        """

        intent_prompt = f"""
        You are a classification engine. Do not explain or speak.
        You're a part of a driver assistant that helps drivers in Cairo, Egypt.

        ONLY return a valid JSON object with EXACTLY 2 elements:
        - type: one of ["chat", "weather", "traffic", "eta", "navigate"]
        - destination: like "Giza", "Nasr City", or leave it empty ("") if user means their current location.

        Examples:
        User: "how’s traffic near me?"
        → {{ "type": "traffic", "destination": "" }}

        User: "navigate to maadi"
        → {{ "type": "navigate", "destination": "maadi" }}

        User: "what's the weather like in Giza?"
        → {{ "type": "weather", "destination": "Giza" }}

        User: "tell me a joke"
        → {{ "type": "chat", "destination": "" }}

        Now classify this input:
        "{query}"
        Respond ONLY with JSON. No explanation.
        """

        try:
            # Send the prompt to the LLM via Ollama
            response = ollama.chat(model="llama3.2", messages=[
                { "role": "user", "content": intent_prompt }
            ])

            # Parse and return the JSON response from the model
            intent_json = json.loads(response['message']['content'])
            return intent_json

        except Exception as e:
            print("[ERROR] Intent classification failed:", e)
            return { "type": "chat", "destination": "" }  # Default fallback


    @eel.expose
    def PassToLlm(self):
        """
        Activates assistant-mode conversation loop.
        
        Continuously:
        - Listens for speech input
        - Tracks silence timeout
        - Routes valid queries to intent handlers
        - Resets back to monitoring mode after inactivity

        This function remains active only while:
            self.state.current_mode == "assistance"
        """
        silence_timeout = 30  # Max duration (in seconds) of silence before exiting
        last_response_time = time.time()  # Timestamp of last valid user input

        while self.state.current_mode == "assistance":
            query = self.User.takecommand()  # Capture user speech (STT)
            print(f"[USER]: {query}")

            # === Silence Handling ===
            if query == "":
                time_since_last = time.time() - last_response_time
                if time_since_last > silence_timeout:
                    eel.DisplayMessage("No response detected. Going back to monitoring mode.")
                    self.Audio.speak("No response detected. Going back to monitoring mode.")
                    self.state.current_mode = "monitoring"
                    self.state.mic_pressed = False
                    eel.ExitHood()
                    break
                continue  # Try listening again if still within timeout window

            # === Valid input received ===
            last_response_time = time.time()  # Reset silence clock

            # Append user query to LLM conversation context
            self.state.conversation_history.append({
                "role": "user",
                "content": query
            })

            




            # === GPS Command Handling (Open / Close) ===

            gps_close_phrases = [
                "close gps", "stop gps", "turn off gps", 
                "hide", "hide gps", "close map", "stop map"
            ]
            gps_open_phrases = [
                "open gps", "turn on gps", "open maps", "open"
            ]

            if any(phrase in query for phrase in gps_close_phrases):
                # --- Close GPS/Maps ---
                from engine.features import CloseMaps
                print("[DEBUG] Closing Google Maps...")
                self.Audio.speak("Got it. Closing maps.")
                eel.DisplayMessage("Got it!")
                eel.DisplayMessage("Closing maps.")
                CloseMaps()
                
                # Reset state to monitoring
                self.state.current_mode = "monitoring"
                self.state.mic_pressed = False
                eel.ExitHood()
                eel.DisplayMessage("")
                break

            elif any(phrase in query for phrase in gps_open_phrases):
                # --- Open GPS/Maps ---
                from engine.features import OpenGps
                print("[DEBUG] Opening Google Maps...")
                self.Audio.speak("Got it. Opening maps.")
                eel.DisplayMessage("Got it!")
                eel.DisplayMessage("Opening maps.")
                OpenGps("gps")
                
                # Reset state to monitoring
                self.state.current_mode = "monitoring"
                self.state.mic_pressed = False
                eel.ExitHood()
                eel.DisplayMessage("")
                break


            # === Exit Assistant Mode (Exit Phrases) ===

            exit_phrases = [
                "goodbye", "bye", "thank you", "thanks", 
                "exit", "end", "close"
            ]

            if any(phrase in query for phrase in exit_phrases):
                print("[DEBUG] Received exit command. Returning to monitoring mode.")

                # Provide user feedback
                farewell = "Goodbye driver."
                eel.DisplayMessage(farewell)
                self.Audio.speak(farewell)

                # Reset assistant state
                self.state.current_mode = "monitoring"
                self.state.mic_pressed = False
                eel.ExitHood()
                eel.DisplayMessage("")
                break


            # === Enable Monitoring Mode (Manual Trigger) ===

            enable_monitoring_phrases = [
                "enable monitoring", "monitoring mode", 
                "back to monitoring", "start monitoring", 
                "start monitor", "enable"
            ]

            if any(phrase in query for phrase in enable_monitoring_phrases):
                print("[DEBUG] User requested to enable monitoring mode.")

                # User feedback
                eel.DisplayMessage("Got it!")
                self.Audio.speak("Got it!")
                eel.DisplayMessage("Switching to monitoring mode.")
                self.Audio.speak("Switching to monitoring mode.")

                # Update assistant state
                self.state.current_mode = "monitoring"
                self.state.mic_pressed = False
                self.state.speak_flag = True
                self.state.json_flag = True
                print(f"[DEBUG] speak_flag set to {self.state.speak_flag}")

                eel.ExitHood()
                eel.DisplayMessage("")
                break



            # === Disable Monitoring Mode (Manual Trigger) ===

            disable_monitoring_phrases = [
                "disable monitoring", "off monitoring", 
                "end monitoring", "disable"
            ]

            if any(phrase in query for phrase in disable_monitoring_phrases):
                print("[DEBUG] User requested to disable monitoring mode.")

                # User feedback
                eel.DisplayMessage("Got it!")
                self.Audio.speak("Got it!")
                disable_msg = (
                    "Monitoring is disabled, but it's a good feature for your safety. "
                    "If you want to re-enable it, just let me know."
                )
                eel.DisplayMessage(disable_msg)
                self.Audio.speak(disable_msg)

                # Update assistant state
                self.state.current_mode = "monitoring"  # still in monitoring mode, but passive
                self.state.mic_pressed = False
                self.state.speak_flag = False
                self.state.json_flag = False
                print(f"[DEBUG] speak_flag set to {self.state.speak_flag}")

                eel.ExitHood()
                eel.DisplayMessage("")
                break

            # === Communication: Send Message or Make Call ===

            communication_keywords = [
                "send", "text", "message", "whatsapp",
                "call", "voice call", "make a call", "ring"
            ]

            if any(phrase in query for phrase in communication_keywords):
                self.state.current_mode = "assistance"

                # Determine if the query is a call or message
                is_call = any(word in query for word in ["call", "voice call", "make a call", "ring"])
                action_type = "call" if is_call else "message"

                # Define your contact list here
                contacts = {
                    "contact_name1": "",  # e.g., "mama": "+201234567890"
                    "contact_name2": "",
                    # Add more contacts
                }

                # Try to extract the contact name from the query
                name = next((c for c in contacts if c in query), None)

                # If name not found in query, ask user to repeat
                if name is None:
                    self.Audio.speak("Sorry, I didn’t catch the name. Can you repeat it?")
                    name_attempt = self.User.takecommand()
                    name = next((c for c in contacts if c in name_attempt), None)

                    if name is None:
                        self.Audio.speak("I still can't catch the name. Exiting.")
                        return  # Abort the action

                number = contacts[name]
                from engine.features import send_whatsApp_msg

                # === CALL HANDLING ===
                if action_type == "call":
                    self.Audio.speak(f"Calling {name} on WhatsApp.")
                    send_whatsApp_msg(number, message="", flag="call", name=name)
                    continue

                # === MESSAGE HANDLING ===
                else:
                    self.Audio.speak(f"Got it. What message should I send to {name}?")
                    eel.DisplayMessage(f"Now tell me the message to send to {name}.")

                    message = "none"
                    timeout = 25  # Max time to wait for user to speak
                    start_time = time.time()
                    retry_prompt_given = False

                    while message == "none" and (time.time() - start_time) < timeout:
                        message = self.User.takecommand()
                        if message == "none" and not retry_prompt_given:
                            self.Audio.speak("Please say it again.")
                            retry_prompt_given = True

                    if message == "none":
                        self.Audio.speak("No message received. Exiting.")
                        return

                    send_whatsApp_msg(number, message, flag="message", name=name)
                    self.Audio.speak("Message sent.")
                    continue


        
            # === Handle Intents (Weather, Navigation, ETA, Traffic, Chat) ===

            intent = self.classify_user_intent(query)
            intent_type = intent.get("type")
            destination = intent.get("destination", "")

            print(f"[DEBUG] Intent type: {intent_type}")
            print(f"[DEBUG] Destination: {destination}")

            # --- ETA or Traffic Info ---
            if intent_type in ["eta", "traffic"]:
                try:
                    # Get coordinates of destination and origin
                    dest_lat, dest_lon = self.geocode_destination(destination)
                    with open("location.json", "r") as f:
                        loc = json.load(f)
                        origin_lat = loc["latitude"]
                        origin_lon = loc["longitude"]

                    # Fetch and speak route info
                    route_response = self.get_route_info(origin_lat, origin_lon, dest_lat, dest_lon)
                    eel.DisplayMessage(route_response)
                    self.Audio.speak(route_response)
                except Exception as e:
                    print("[ERROR] Route information failed:", e)
                    self.Audio.speak("Sorry, I couldn't retrieve route information.")
                continue

            # --- Navigation Intent ---
            elif intent_type == "navigate":
                try:
                    self.handle_navigation(destination)
                except Exception as e:
                    print("[ERROR] Navigation handling failed:", e)
                    self.Audio.speak("Sorry, I couldn't start navigation.")
                continue

            # --- Chat Intent (General conversation) ---
            elif intent_type == "chat":
                try:
                    user_msg = { "role": "user", "content": query }
                    self.state.conversation_history.append(user_msg)

                    # Trim history to keep token limits
                    trimmed_history = self.trim_history(self.state.conversation_history)
                    chat_response = ollama.chat(model="llama3.2", messages=trimmed_history)
                    reply = chat_response["message"]["content"]

                    # Respond and store
                    eel.DisplayMessage(reply)
                    self.Audio.speak(reply)
                    self.state.conversation_history.append({ "role": "assistant", "content": reply })
                except (requests.exceptions.RequestException, KeyError, ValueError, TypeError) as e:
                    print(f"[ERROR] Chat LLM call failed: {e}")
                    eel.DisplayMessage("Sorry, I couldn't process that.")
                    self.Audio.speak("Sorry, I couldn't process that.")
                continue

            # --- Weather Intent ---
            elif intent_type == "weather":
                try:
                    # If user asked about a specific destination
                    if destination:
                        lat, lon = self.geocode_destination(destination)
                    else:
                        # Otherwise, get current location from file
                        with open("location.json", "r") as f:
                            loc = json.load(f)
                            lat = loc["latitude"]
                            lon = loc["longitude"]

                    weather = self.get_weather(lat=lat, lon=lon)
                    eel.DisplayMessage(weather)
                    self.Audio.speak(weather)
                except (requests.exceptions.RequestException, KeyError, ValueError) as e:
                    print("[ERROR] Weather handling failed:", e)
                    self.Audio.speak("Sorry, I couldn't fetch the weather right now.")
                continue



                    
    def geocode_destination(self, destination):
        """
        Converts a user-spoken destination into geographic coordinates (latitude, longitude)
        using Google Maps Places API (Autocomplete + Details).
        """
        # --- Step 1: Use Autocomplete API to get the most relevant place suggestion ---
        autocomplete_url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params_autocomplete = {
            "input": destination,
            #"key": GOOGLE_MAPS_KEY,
            "language": "en",
            "region": "eg"
        }

        try:
            resp = requests.get(autocomplete_url, params=params_autocomplete, timeout=5).json()
        except Exception as e:
            print(f"[ERROR] Autocomplete API call failed: {e}")
            self.Audio.speak("There was a problem reaching the map service.")
            return None, None

        # If no predictions returned or request failed
        if resp.get("status") != "OK" or not resp.get("predictions"):
            print(f"[ERROR] Autocomplete failed: {resp.get('status')}")
            self.Audio.speak("Sorry, I couldn't hear you clearly. Can you repeat?")
            return self.PassToLlm()

        # --- Clean up raw description for logging/debugging purposes ---
        raw_description = resp["predictions"][0]["description"]
        ascii_only = re.sub(r'[^\x00-\x7F]+', '', raw_description)  # Remove emojis, Arabic, etc.
        clean_description = re.sub(r'\s*,\s*', ', ', ascii_only)
        clean_description = re.sub(r'(,\s*)+', ', ', clean_description)
        clean_description = re.sub(r'\s+', ' ', clean_description).strip()

        print(f"[DEBUG] Top match: {clean_description}")

        # Extract place_id for detailed lookup
        place_id = resp["predictions"][0]["place_id"]

        # --- Step 2: Get place details to extract latitude/longitude ---
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        params_details = {
            "place_id": place_id,
             #"key": GOOGLE_MAPS_KEY,
            "language": "en"
        }

        try:
            details_resp = requests.get(details_url, params=params_details, timeout=5).json()
        except Exception as e:
            print(f"[ERROR] Place Details API call failed: {e}")
            self.Audio.speak("There was a problem getting location details.")
            return None, None

        if details_resp.get("status") != "OK":
            print(f"[ERROR] Place details failed: {details_resp.get('status')}")
            self.Audio.speak("Sorry, I couldn’t find that place.")
            return None, None

        location = details_resp["result"]["geometry"]["location"]
        dest_lat = location["lat"]
        dest_lon = location["lng"]

        print(f"[DEBUG] Destination coordinates: {dest_lat}, {dest_lon}")
        return dest_lat, dest_lon



    def handle_navigation(self, destination):
        """
        Uses Google Maps APIs to process the destination, fetch coordinates, 
        and open navigation directions in the browser.
        """
        print(f"[DEBUG] Handling navigation to: {destination}")

        # --- Step 1: Use Google Autocomplete API to resolve user query into a place description ---
        autocomplete_url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params_autocomplete = {
            "input": destination,
            #"key": GOOGLE_MAPS_KEY,
            "language": "en",
            "region": "eg"
        }

        try:
            resp = requests.get(autocomplete_url, params=params_autocomplete, timeout=5).json()
        except Exception as e:
            print(f"[ERROR] Autocomplete API call failed: {e}")
            self.Audio.speak("There was a problem reaching the map service.")
            return

        if resp.get("status") != "OK" or not resp.get("predictions"):
            print(f"[ERROR] Autocomplete failed: {resp.get('status')}")
            self.Audio.speak("Sorry, I couldn't hear you clearly. Can you repeat?")
            return self.PassToLlm()  # Optionally handle fallback in LLM

        raw_description = resp['predictions'][0]['description']

        # --- Step 2: Clean up the location string ---
        ascii_only = re.sub(r'[^\x00-\x7F]+', '', raw_description)  # Remove emojis, Arabic, etc.
        clean_description = re.sub(r'\s*,\s*', ', ', ascii_only)    # Normalize commas
        clean_description = re.sub(r'(,\s*)+', ', ', clean_description)
        clean_description = re.sub(r'\s+', ' ', clean_description).strip()

        print(f"[DEBUG] Top match: {clean_description}")

        if not clean_description:
            self.Audio.speak("Where would you like to go?")
            return

        # --- Step 3: Speak destination feedback to user ---
        eel.DisplayMessage(f"Opening directions to {clean_description.title()}...")
        self.Audio.speak(f"Opening directions to {clean_description.title()}...")

        # --- Step 4: Use Google Place Details API to get lat/lon ---
        dest_lat, dest_lon = self.geocode_destination(destination)
        if dest_lat is None or dest_lon is None:
            self.Audio.speak("Sorry, I couldn't find that location.")
            return

        # --- Step 5: Open Google Maps directions in browser ---
        map_url = f"https://www.google.com/maps/dir/?api=1&destination={dest_lat},{dest_lon}"
        webbrowser.open(map_url)


    def get_route_info(self, origin_lat, origin_lon, dest_lat, dest_lon):
        """
        Gets step-by-step driving directions and travel time between origin and destination,
        summarizes the route using an LLM for voice-friendly feedback.
        """
        # --- Validate coordinates ---
        if not all([origin_lat, origin_lon, dest_lat, dest_lon]):
            print("[ERROR] Invalid coordinates provided.")
            return "Location data is incomplete."

        # --- Google Directions API request ---
        directions_url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            "origin": f"{origin_lat},{origin_lon}",
            "destination": f"{dest_lat},{dest_lon}",
            "departure_time": "now",  # Uses real-time traffic
            #"key": GOOGLE_MAPS_KEY
        }

        try:
            response = requests.get(directions_url, params=params, timeout=5).json()
        except Exception as e:
            print(f"[ERROR] Failed to fetch directions: {e}")
            return "Unable to connect to the maps service."

        if response.get("status") != "OK":
            print(f"[ERROR] Directions failed: {response.get('status')}")
            return "Could not retrieve directions."

        # --- Extract route details ---
        route = response["routes"][0]["legs"][0]
        duration = route["duration"]["text"]
        duration_sec = route["duration"]["value"]
        traffic_sec = route.get("duration_in_traffic", {}).get("value", duration_sec)
        duration_traffic_text = route.get("duration_in_traffic", {}).get("text", duration)
        distance = route["distance"]["text"]

        # --- Basic summary ---
        output = [f"Estimated travel time is {duration_traffic_text}, covering a distance of {distance}."]

        # --- Add traffic delay note if significant ---
        delay_sec = traffic_sec - duration_sec
        if delay_sec > 60:
            delay_min = round(delay_sec / 60)
            output.append(f"Note: Expect a delay of approximately {delay_min} minute{'s' if delay_min > 1 else ''} due to traffic.")

        # --- Step-by-step route instructions ---
        output.append("Here are the main steps:")
        for i, step in enumerate(route['steps'], 1):
            # Strip HTML, Arabic text, and clean spacing
            instruction = re.sub('<[^<]+?>', '', step['html_instructions'])  # Remove HTML
            instruction = re.sub(r'[\u0600-\u06FF]+', '', instruction)       # Remove Arabic text
            instruction = re.sub(r'\s+', ' ', instruction).strip()
            step_distance = step['distance']['text']
            step_duration = step['duration']['text']
            output.append(f"{i}. {instruction} ({step_distance}, {step_duration})")

        full_route_text = " ".join(output)

        # --- Ask LLM to summarize for driver ---
        user_msg = {
            "role": "user",
            "content": f"Summarize this driving route in 3 to 4 short sentences for speaking out loud, and include the estimated travel time and distance:\n{full_route_text}"
        }
        self.state.conversation_history.append(user_msg)

        try:
            trimmed = self.trim_history(self.state.conversation_history)
            chat_response = ollama.chat(model="llama3.2", messages=trimmed)
            reply = chat_response["message"]["content"]
            self.state.conversation_history.append({ "role": "assistant", "content": reply })
            return reply  # Return summarized version
        except (requests.exceptions.RequestException, KeyError, ValueError, TypeError) as e:
            print(f"[LLM ERROR] Could not summarize route: {e}")
            return full_route_text  # Fallback to full text



        
    def get_weather(self, lat=None, lon=None, location_name=None):
        """
        Fetches current weather data from the OpenWeatherMap API using either:
        - Latitude and longitude (preferred)
        - Location name (fallback)

        Returns a spoken-style summary of current conditions.
        """
        
        #OPENWEATHER_KEY = WEATHER_KEY  # Use environment variable in production

        # --- Build API URL based on input ---
        if lat and lon:
            #url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_KEY}&units=metric"
        elif location_name:
            #url = f"https://api.openweathermap.org/data/2.5/weather?q={location_name}&appid={OPENWEATHER_KEY}&units=metric"
        else:
            return "Location for weather not provided."

        try:
            # --- Request weather data ---
            response = requests.get(url)
            data = response.json()

            # --- Check for API errors ---
            if data.get("cod") != 200:
                print(f"[ERROR] Weather API error: {data.get('message')}")
                return "Sorry, I couldn't fetch the weather data."

            # --- Extract relevant weather info ---
            description = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            city = data["name"]

            # --- Generate summary report ---
            weather_report = (
                f"Current weather in {city} is {description}. "
                f"Temperature is {temp}°C, feels like {feels_like}°C. "
                f"Humidity is {humidity}%."
            )
            return weather_report

        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print("[ERROR] Weather API call failed:", e)
            return "Weather data is currently unavailable."


# =========================
# USER MANAGER CLASS
# ========================= 
   
class UserManager:
    def __init__(self, state, Audio):
        """
        Initializes the UserManager with app state and audio manager references.
        
        Args:
            state (AppState): Global app state object to track current mode, flags, etc.
            Audio (AudioManager): Instance to handle TTS and other audio playback.
        """
        self.state = state
        self.Audio = Audio
        #self.client = Groq(api_key=GROQ_KEY)  # Initialize Groq client for transcription

    def record_audio(self, duration=6, sample_rate=16000, file_path="input.wav"):
        """
        Records audio from the microphone and saves it as a WAV file.

        Args:
            duration (int): Duration to record in seconds (default: 6).
            sample_rate (int): Sampling rate for audio capture (default: 16000 Hz).
            file_path (str): Path to save the recorded file.

        Returns:
            str: Path to the saved audio file.
        """
        import sounddevice as sd
        from scipy.io.wavfile import write

        print("[🎙️] Recording audio...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        sd.wait()
        write(file_path, sample_rate, audio)
        print(f"[✅] Audio saved to {file_path}")
        return file_path

    def is_audio_silent(self, filepath, threshold_db=-40.0):
        """
        Checks if the audio file has low energy (silent or near-silent).

        Args:
            filepath (str): Path to the WAV file.
            threshold_db (float): Silence threshold in decibels relative to full scale.

        Returns:
            bool: True if audio is considered silent, False otherwise.
        """
        audio = AudioSegment.from_file(filepath)
        return audio.dBFS < threshold_db

    def transcribe_audio_with_groq(self, file_path, lang="en"):
        """
        Transcribes the recorded audio using Groq's Whisper Large v3 API.

        Args:
            file_path (str): Path to the WAV file.
            lang (str): Language code (default: English "en").

        Returns:
            str: Transcribed text in lowercase or "none" if an error occurs or audio is silent.
        """
        # Custom prompt tailored for Egyptian voice commands in a driving context
        prompt = (
            "You are transcribing short English voice commands from a driver assistant system in Egypt. "
            "Expect navigation-related words like 'navigate to', 'how far is', and 'estimated arrival time'. "
            "Egyptian area names may include: Zahraa El Maadi, Maadi, Nasr City, Dokki, Mohandessin, Giza, Zamalek, Sheraton, "
            "New Cairo, El Rehab, Shorouk, 6 October, Giza Pyramids, Sheikh Zayed, Heliopolis, Downtown Cairo, Cairo Airport, Mokattam, "
            "Garden City, Manial, Abbasia, Ramses, Sayeda Zeinab, Helwan, Ain Shams, Hadayek El Maadi, El Marg, El Obour, El Salam, "
            "El Basatin, El Tagamoa, Fifth Settlement, Sixth Settlement, Masr El Gedida, Badr City, El Manyal, El Matareya, "
            "El Waily, El Darassa, El Mokattam, Tora, Zahraa Nasr City, Gesr El Suez, New Capital, El Talbia, Warraq, Kerdasa, "
            "Boulaq, Faisal Street, Haram, Imbaba, Rod El Farag, El Zawya El Hamra, El Sayeda Aisha, and El Mohandessin."
        )

        try:
            # Skip transcription if input audio is silent
            if self.is_audio_silent(file_path):
                print("[⚠️] Skipping transcription: audio is silent or low energy.")
                return ""

            # Open file and send to Whisper API via Groq
            with open(file_path, "rb") as file:
                result = self.client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3",
                    language=lang,
                    response_format="text",
                    temperature=0.2,
                    prompt=prompt,
                )

            return result.strip().lower()

        except Exception as e:
            print(f"[❌] Whisper API error: {e}")
            return "none"

        finally:
            # Clean up temporary audio file
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[🗑️] Deleted {file_path}")


    @eel.expose
    def takecommand(self, timeout=15, phrase_time_limit=7, lang="en"):
        """
        Records and transcribes a short voice command using the microphone.

        Args:
            timeout (int): Max duration to wait before auto-stopping (unused in current logic).
            phrase_time_limit (int): Max audio duration in seconds to record.
            lang (str): Language code for transcription.

        Returns:
            str: Lowercase recognized text from audio or "none" on failure.
        """
        try:
            print("[🎙️] Listening for command...")
            eel.DisplayMessage("[Listening for command...]")

            # Record audio for the given time limit
            file_path = self.record_audio(duration=phrase_time_limit)

            print("[Recognizing...]")
            eel.DisplayMessage("[Recognizing...]")

            # Transcribe the recorded audio
            recognized_text = self.transcribe_audio_with_groq(file_path, lang=lang)

            print(f"[STT] You said: {recognized_text}")
            return recognized_text

        except Exception as e:
            print("[ERROR] takecommand failed:", e)
            return "none"


    @eel.expose
    def ListenForWakeWord(self, phrase_time_limit=5, wake_word="hey nova"):
        """
        Listens for a predefined wake word (e.g., "hey nova") from user's voice.

        Args:
            phrase_time_limit (int): Duration to listen for the wake word.
            wake_word (str): The trigger word to detect.

        Returns:
            bool: True if wake word is detected, False otherwise.
        """
        print("[Listening for wake word...]")

        # Record short audio to check for wake word
        file_path = self.record_audio(duration=phrase_time_limit)

        print("[Recognizing...]")
        recognized_text = self.transcribe_audio_with_groq(file_path, lang="en")
        print(f"[STT] You said: {recognized_text}")

        if wake_word in recognized_text:
            # Wake word matched, trigger assistant response
            eel.DisplayMessage("Hey driver, how can I help you?")
            self.Audio.speak("Hey driver, how can I help you?")
            time.sleep(0.5)
            return True

        return False

            
    def send_feedback_to_EC(self):
        """
        Triggers an emergency protocol by notifying the predefined emergency contact via:
        1. Email (with a livestream link)
        2. WhatsApp message (with location + stream info)
        3. WhatsApp call (for urgent voice contact)
        """

        # === Step 1: Send alert email with live stream link ===
        send_alert_email(LIVE_STREAM_URL, RECEIVER_EMAIL)

        # === Step 2: Load current location from local JSON file ===
        try:
            with open('location.json', 'r') as f:
                data = json.load(f)
                lat = data.get("latitude")
                lon = data.get("longitude")
                address = data.get("address", "Unknown location")
        except Exception as e:
            print(f"[ERROR] Failed to read location: {e}")
            lat, lon, address = None, None, "Unknown location"

        # === Step 3: Generate Google Maps link ===
        map_url = f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else "#"

        # === Step 4: Send WhatsApp Message with Location and Live Stream ===
        from engine.features import send_whatsApp_msg
        message = (
            f"🚨 *Emergency Alert!* 🚨\n"
            f"Driver seems to be unresponsive.\n\n"
            f"🗺️ *Location:* {map_url}\n"
            f"📹 *Live Stream:* {LIVE_STREAM_URL}\n"
        )

        send_whatsApp_msg(
            EMERGENCY_CONTACT_NUMBER,
            message,
            flag='message',
            name=EMERGENCY_CONTACT_NAME
        )

        # === Step 5: Initiate WhatsApp Voice Call ===
        time.sleep(1)  # short pause before call
        send_whatsApp_msg(
            EMERGENCY_CONTACT_NUMBER,
            message="",
            flag='call',
            name=EMERGENCY_CONTACT_NAME
        )


    def alert(self):
        """
        Triggered when fatigue is detected.
        - First, tries twice to confirm if the driver is responsive.
        - If both checks fail, it alerts the emergency contact via email and WhatsApp.
        - If the driver responds, asks if they’d like to talk, and switches to assistance mode accordingly.
        """
        self.state.current_mode = "sleep_alert"

        # === First Prompt ===
        eel.DisplayMessage("Are you okay? Can you hear me?")
        self.Audio.speak("Are you okay? Can you hear me?")
        eel.ShowHood()
        time.sleep(0.5)

        # Wait for first response
        response_1 = self.check_up(timeout=10)
        eel.ExitHood()
        time.sleep(2)  # Pause before retry

        # === Second Prompt if No Response ===
        response_2 = None
        if response_1 is None:
            self.state.current_mode = "sleep_alert"
            eel.DisplayMessage("Are you okay? Can you hear me?")
            self.Audio.speak("Are you okay? Can you hear me?")
            eel.ShowHood()
            time.sleep(0.5)

            # Wait for second response
            response_2 = self.check_up(timeout=10)

        # === Final Check: No Response from Both Prompts ===
        if response_1 is None and response_2 is None:
            eel.DisplayMessage("Dangerous! No response from driver.")
            self.Audio.speak("Dangerous! No response from driver.")
            self.Audio.BuzzerSound()
            time.sleep(0.5)
            self.send_feedback_to_EC()  # Notify emergency contact
            time.sleep(7)
            self.state.current_mode = "monitoring"
            eel.ExitHood()
            return

        # === Driver Responded ===
        eel.DisplayMessage("Do you want to talk to me?")
        self.Audio.speak("Do you want to talk to me?")
        time.sleep(0.5)

        answer = self.check_up(timeout=10)

        if answer and "yes" in answer:
            eel.DisplayMessage("Okay, I'm here to help you, you're in assistance mode now.")
            self.Audio.speak("Okay, I'm here to help you, you're in assistance mode now.")
            eel.ShowHood()
            self.state.current_mode = "assistance"
        else:
            eel.DisplayMessage("Okay, let me know if you need me.")
            self.Audio.speak("Okay, let me know if you need me.")
            self.state.current_mode = "monitoring"
            eel.ExitHood()


    def check_up(self, timeout=10, lang="en"):
        """
        Listens for any vocal response from the user within the specified timeout.
        
        Args:
            timeout (int): Duration in seconds to wait for a voice input.
            lang (str): Language code for speech-to-text processing.

        Returns:
            str | None: Transcribed user response in lowercase if heard, otherwise None.
        """
        try:
            print("[🟡] Listening for response...")
            eel.DisplayMessage("[Listening for response...]")

            # Record audio input from user
            file_path = self.record_audio(duration=timeout, file_path="response.wav")

            # Transcribe the recorded audio using Groq Whisper model
            response = self.transcribe_audio_with_groq(file_path, lang=lang)

            # If response detected, return it
            if response:
                print(f"[🗣️] Heard: {response}")
                return response
            else:
                print("[⚠️] No valid response detected.")
                return None

        except Exception as e:
            print(f"[❌] check_up() failed with error: {e}")
            return None



# ==========================================================================
# 🔻🔻🔻🔻 APPLICATION INITIALIZATION – CREATE CLASS INSTANCES 🔻🔻🔻🔻
# ==========================================================================

state = AppState()                   # Global state handler for tracking app mode, context, etc.
Audio = AudioManager()               # Manages TTS, alerts, and audio playback
User = UserManager(state, Audio)     # Handles audio input, wake word detection, and user transcription
LLM = LLMManager(state, Audio, User) # Main logic engine for intent handling, chat, nav, alerts


@eel.expose
def ReceiveLocation(lat, lon):
    """
    Receives latitude and longitude from the frontend and reverse-geocodes the coordinates.
    Saves full location info into a JSON file and updates assistant context.
    """
    try:
        geolocator = Nominatim(user_agent="driver_assistant")
        location = geolocator.reverse((float(lat), float(lon)), language="en")
        address = location.address if location else "Unknown"

        # Save the location data
        with open("location.json", "w") as f:
            json.dump({
                "latitude": lat,
                "longitude": lon,
                "address": address
            }, f)

        print(f"[📍] Precise location: {address}")
        state.location_override = address

        # Update system message in LLM context
        if state.conversation_history and state.conversation_history[0]["role"] == "system":
            state.conversation_history[0] = LLM.generate_initial_context()
        else:
            print("[WARN] System message not found in conversation history.")

    except Exception as e:
        print(f"[ERROR] Failed to reverse geocode location: {e}")



@eel.expose
def set_mic_pressed():
    """
    Triggered from frontend when mic button is manually pressed.
    """
    state.mic_pressed = True

# ========================== Manual Monitoring Control ==========================

@eel.expose
def enable_monitoring():
    """
    Enables monitoring mode manually via frontend (e.g., toggle switch).
    """
    state.json_flag = True
    state.speak_flag = True
    Audio.speak("Monitoring is enabled.")

@eel.expose
def disable_monitoring():
    """
    Disables monitoring mode manually via frontend.
    """
    state.json_flag = False
    Audio.speak("Monitoring is disabled.")

@eel.expose
def get_monitor_mode():
    """
    Returns the current monitoring status as 'on' or 'off' for UI display.
    """
    return "on" if state.json_flag else "off"

def fetch_json_file():
    """
    Downloads the driver alert JSON file from the remote server and saves it locally.

    Source: Remote live stream or monitoring backend (e.g., Jetson).
    Destination: Local path defined in AppState (state.json_file_path).
    """
    try:
        url = "https://bbd7-156-215-169-198.ngrok-free.app/driver_assistant.json"
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()

        # Save the fetched JSON data to a local file for real-time monitoring
        with open(state.json_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print("[✅] JSON file fetched and saved successfully.")

    except requests.exceptions.RequestException as e:
        print(f"[❌] Network or HTTP error while fetching JSON: {e}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[❌] Failed to decode JSON response: {e}")
    except Exception as e:
        print(f"[❌] Unexpected error while fetching JSON: {e}")


@eel.expose
def monitoring_loop():
    """
    Background loop for monitoring mode:
    - Reads alerts from remote JSON
    - Triggers alerts for fatigue or hands-off
    - Detects wake word or manual mic press
    - Sends feedback or suggestions via LLM if issues are detected
    """
    global counter_hands

    while True:
        data = {}  # Reset data each loop

        # Step 1: Fetch latest driver alert data from remote server
        #fetch_json_file()

        # Step 2: Load the data if the file exists
        if os.path.exists(state.json_file_path):
            try:
                with open(state.json_file_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERROR] Could not load or delete JSON file: {e}")

        # Step 3: Fatigue emergency — trigger sleep alert immediately
        if data.get("sleep_alert", "").lower() == "on":
            state.current_mode = "sleep_Alert"
            User.alert()
            time.sleep(5)
            continue

        # Step 4: Track hands on/off wheel status
        hands = data.get("HOW_Alert", "").lower()

        if hands == "hands off wheel":
            counter_hands += 1
            print(f"[DEBUG] Counter of Hands Off wheel: {counter_hands}")
        elif hands == "hands on wheel":
            counter_hands = 0
            print(f"[DEBUG] Counter reset (Hands On): {counter_hands}")

        # Step 5: If hands off counter exceeds threshold → treat as emergency
        if counter_hands > HOW_THRESHOLD:
            state.current_mode = "sleep_Alert"
            User.alert()
            time.sleep(5)
            continue

        # Step 6: Wake word or mic trigger → switch to assistant mode
        if state.mic_pressed or (state.current_mode == "monitoring" and User.ListenForWakeWord()):
            eel.DisplayMessage("")
            state.current_mode = "assistance"
            eel.ShowHood()
            LLM.PassToLlm()
            continue

        # Step 7: Continue assistance if already active
        if state.current_mode == "assistance":
            eel.ShowHood()
            LLM.PassToLlm()
            continue

        # Step 8: Monitoring is active, no mic press, flags are on
        if state.current_mode == "monitoring" and not state.mic_pressed and state.json_flag and state.speak_flag:
            print(f"[DEBUG] json_flag: {state.json_flag}, speak_flag: {state.speak_flag}")

            #fetch_json_file()  # Refetch for freshest data

            if os.path.exists(state.json_file_path):
                with open(state.json_file_path, "r") as f:
                    data = json.load(f)

                    # Stop if mode changed mid-read
                    if state.current_mode != "monitoring":
                        continue

                    # Parse alerts
                    fatigue = data.get("Fatigue_Alert", "").lower()
                    distraction = data.get("Distraction_Alert", "").lower()
                    activity = data.get("Activity_Alert", "").lower()

                    # Skip if everything is fine
                    if activity == "safe driving" and distraction == "off" and hands == "hands on wheel" and fatigue == "off":
                        continue

                    # Step 9: Build a safety instruction prompt for LLM
                    prompt = (
                        f"Driver activity: {activity}. "
                        f"Distraction alert: {distraction}. "
                        f"Hands on or off wheel: {hands}. "
                        f"Fatigue alert: {fatigue}. "
                        f"Based on these observations, provide a short, polite safety instruction in 20 words or less. "
                        f"The tone should be clear and supportive. Avoid generic advice."
                    )

                    try:
                        safe_history = [{"role": "user", "content": prompt}]
                        response = ollama.chat(model='llama3.2', messages=safe_history)
                        reply = response['message']['content']

                        print(f"[LLM] Assistant response: {reply}")
                        Audio.speak(reply)

                    except Exception as e:
                        print(f"[ERROR] LLM failed during monitoring: {e}")
                        eel.DisplayMessage("Error analyzing driver alert data.")
                        Audio.speak("Error analyzing driver alert data.")
        else:
            # State mismatch or flag off → do nothing but wait
            print("[DEBUG] Monitoring skipped due to flag or state mismatch.")
            print(f"[DEBUG] json_flag: {state.json_flag}, speak_flag: {state.speak_flag}")
            continue

        # Step 10: Sleep briefly before next check
        time.sleep(0.1)
