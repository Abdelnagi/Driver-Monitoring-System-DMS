import RPi.GPIO as GPIO
import time
from Display_Images import screen_image

# Pin config (BCM mode)
TRIG = 4
ECHO = 17
HOLD_SEC = 2
UNSAFE_DIST = 200 	# 200 cm

GO_IMAGE = "./images/Go_Sign_rot.png"
WARN_R_IMAGE = "./images/warn_R_rot.png"
WARN_L_IMAGE = "./images/warn_L_rot.png"

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

def get_distance():
    # Ensure trigger is low
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    # Send 10µs pulse
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    # Wait for echo start
    timeout_start = time.time()
    while GPIO.input(ECHO) == 0:
        if time.time() - timeout_start > 0.03:
            return None  # Timeout

    start_time = time.time()

    # Wait for echo end
    timeout_start = time.time()
    while GPIO.input(ECHO) == 1:
        if time.time() - timeout_start > 0.03:
            return None  # Timeout

    end_time = time.time()

    # Calculate distance (in cm)
    duration = end_time - start_time
    distance = (duration * 34300) / 2
    return round(distance,2)

try:
    while True:
        dist = get_distance()
        if dist is None:
            print("No echo")
            screen_image(GO_IMAGE)

        elif dist > UNSAFE_DIST:	# More than 1 meters
            screen_image(GO_IMAGE)
            print(f"Safe: Distance is {dist:.2f} cm")
        elif dist < UNSAFE_DIST:	# Less than 1 meters
            screen_image(WARN_IMAGE)
            print(f"Warning: Distance: {dist:.2f} cm")
        time.sleep(.75)

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    GPIO.cleanup()

