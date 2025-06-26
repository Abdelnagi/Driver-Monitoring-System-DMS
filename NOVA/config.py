"""
Global Configuration Settings

This file stores all sensitive and adjustable parameters used across the project.
It centralizes credentials, keys, contact info, and important system thresholds.

WARNING:
- Do NOT hardcode sensitive keys when deploying.
- Use environment variables or encrypted secrets in production.
"""

# === EMAIL Configuration ===
SENDER_EMAIL = ""            # Email address that will send alerts or updates
APP_PASSWORD = ""            # App-specific password or token for the sender email
RECEIVER_EMAIL = ""          # Email address to receive system notifications

# === Emergency Contact Configuration ===
EMERGENCY_CONTACT_NUMBER = ""   # Phone number of the emergency contact person
EMERGENCY_CONTACT_NAME = ""     # Full name of the emergency contact
LIVE_STREAM_URL = ""            # Public or internal URL to the driver’s live camera feed

# === API Keys ===
WEATHER_API_KEY = ""        # API key for fetching weather data (e.g., OpenWeatherMap)
LOCATION_API_KEY = ""       # API key for geolocation services
GOOGLE_MAPS_API_KEY = ""    # API key for Google Maps integration (e.g., routing/navigation)
GROQ_API_KEY = ""           # API key for whisper STT - for the Groq inference service (LLM or other models)
HERE_API_KEY = ""           # API key for HERE Maps or HERE location services



# === System Thresholds ===
HOW_THRESHOLD = 2           # Hands-Off-Wheel alert threshold (e.g., number of seconds or detections)
