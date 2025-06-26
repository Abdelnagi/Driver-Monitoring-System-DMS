"""
@file send.py
@brief Sends a styled HTML emergency alert email with live stream and location info.
"""

# === IMPORTS ===
import smtplib
from email.message import EmailMessage
import requests
import pywifi
import time
import json
from config import *  # Contains all necessary API keys, email


# === LOCATION PROVIDER CLASS ===

class LocationProvider:
    """
    Handles location fetching using locally saved JSON or nearby Wi-Fi networks (future use).
    """

    def __init__(self):
        self.api_key = LOCATION_API_KEY

    def scan_wifi_windows(self):
        """
        Optional: Scan nearby Wi-Fi networks using pywifi (Windows only).
        Returns: List of dicts with MAC and signal strength.
        """
        wifi = pywifi.PyWiFi()
        iface = wifi.interfaces()[0]

        iface.scan()
        time.sleep(3)  # Wait for scan to complete
        results = iface.scan_results()

        wlan_data = []
        for network in results:
            wlan_data.append({
                "mac": network.bssid,
                "signalStrength": network.signal
            })

        return wlan_data

    def get_current_location(self):
        """
        Load current location from a local JSON file.
        Expects: 'location.json' with keys: latitude, longitude, address.
        Returns: lat (float), lon (float), address (str), Google Maps URL (str)
        """
        try:
            with open('location.json', 'r') as f:
                data = json.load(f)

            lat = float(data.get("latitude"))
            lon = float(data.get("longitude"))
            address = data.get("address", "Unknown location")
            map_url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

            return lat, lon, address, map_url

        except (ValueError, TypeError) as ve:
            print("⚠ Invalid location values:", ve)
            return None, None, "Invalid location data", "#"

        except Exception as e:
            print("⚠ Failed to get location:", e)
            return None, None, "Location unavailable", "#"


# === EMAIL ALERT FUNCTION ===

def send_alert_email(url: str, to_email: str):
    """
    Sends an HTML-formatted emergency alert email containing:
    - Driver's last known location
    - Link to Google Maps
    - Link to live camera stream

    Args:
        url (str): Live stream URL to include in plain-text fallback
        to_email (str): Receiver email address
    """

    # Retrieve current location data
    location_provider = LocationProvider()
    lat, lon, address, map_url = location_provider.get_current_location()

    print(f"[DEBUG] Retrieved location: lat={lat}, lon={lon}")

    # Fallback values if location is unavailable
    if lat is not None and lon is not None:
        location = f"{lat:.6f}, {lon:.6f}"
        map_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
    else:
        location = "Unknown"
        map_url = "https://www.google.com/maps"

    # Create the email message
    msg = EmailMessage()
    msg['Subject'] = '🚨 Emergency Alert: Driver Unresponsive'
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email

    # Plain-text version (fallback)
    msg.set_content(f'''
    ALERT! The driver may be asleep.
    Location: {location}
    Stream: {url}
    Map: {map_url}
    ''')

    # HTML-styled version
    html_content = f"""
    <html>
    <body style="margin: 0; padding: 0; background-color: #f9f9f9; font-family: 'Segoe UI', sans-serif;">
        <div style="max-width: 600px; margin: 40px auto; background: #ffffff; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); overflow: hidden;">
            <div style="background: #1a1a2e; padding: 24px; text-align: center;">
                <h2 style="color: #ffffff; margin: 0;">🚨 Driver Safety Alert</h2>
            </div>

            <div style="padding: 30px;">
                <p style="font-size: 17px; color: #222;">
                    <strong>Alert:</strong> The driver appears to be <span style="color: #e63946;"><strong>unresponsive</strong></span>.
                </p>

                <p style="font-size: 15px; color: #444;">
                    <strong>Last known location:</strong> <span style="color: #1a8cff;">{location}</span>
                </p>

                <div style="text-align: center; margin: 20px 0 30px 0;">
                    <a href="{map_url}" target="_blank"
                    style="background: #ffc107; color: black; padding: 12px 24px; text-decoration: none; font-weight: bold; font-size: 16px; border-radius: 30px;">
                    📍 Live Location
                    </a>
                </div>

                <div style="text-align: center; margin: 20px 0;">
                    <a href="{LIVE_STREAM_URL}" target="_blank"
                    style="background: #1a8cff; color: white; padding: 14px 28px; text-decoration: none; font-weight: bold; font-size: 16px; border-radius: 30px;">
                    ▶️ Live Streaming
                    </a>
                </div>

                <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 30px 0;">
                <p style="font-size: 13px; color: #888; text-align: center;">
                    Sent automatically by your Driver Monitoring System.<br>
                    This is an automated alert — please do not reply.
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    msg.add_alternative(html_content, subtype='html')

    # Send the email via secure SMTP
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
            print("✅ Alert email sent successfully!")
    except Exception as e:
        print("❌ Failed to send email:", e)
