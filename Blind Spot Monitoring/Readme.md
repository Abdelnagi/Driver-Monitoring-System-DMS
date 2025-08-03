# Raspberry Pi Blind Spot Monitor

This project implements a Blind Spot Monitoring (BSM) system using a Raspberry Pi, an ultrasonic sensor, and an LCD screen. The system detects objects in the vehicle's blind spot and displays a visual warning.

---

## How it Works

The `BSM.py` script uses an ultrasonic sensor to measure the distance to nearby objects.

* If an object is detected within a predefined "unsafe" distance, a warning image is displayed on the screen.
* If no object is detected or the object is at a safe distance, a "go" image is displayed.

The `Display_Images.py` script handles the logic for displaying images on the ST7735 TFT LCD screen.

---

## Required Libraries & Installation

### BSM.py

This script is responsible for the main logic of the blind spot monitoring system.

* **RPi.GPIO**: This library is used to control the GPIO pins of the Raspberry Pi to interact with the ultrasonic sensor.
* **time**: A standard Python library for handling time-related tasks, such as delays.

### Display_Images.py

This script controls the TFT LCD screen.

* **PIL (Pillow)**: The Python Imaging Library is used for opening, manipulating, and displaying images.
* **ST7735**: This library is required to interface with the ST7735 TFT LCD controller.
* **Adafruit_GPIO**: This library is a dependency for the ST7735 driver to handle SPI communication.

### Installation Command

You can install the necessary libraries by running the following command in your terminal:

```bash
pip install RPi.GPIO pillow adafruit-pureio adafruit-circuitpython-st7735