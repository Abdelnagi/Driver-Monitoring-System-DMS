from PIL import Image
import time
import ST7735 as TFT
import Adafruit_GPIO.SPI as SPI

WIDTH = 128
HEIGHT = 160
ANGLE = -90
SPEED_HZ = 40000

# Raspberry Pi configuration.
DC = 24
RST = 25
SPI_PORT = 0
SPI_DEVICE = 0

# Create TFT LCD display class.
disp = TFT.ST7735(
    DC,
    width=WIDTH, height=HEIGHT,
    rst=RST,
    spi=SPI.SpiDev(
        SPI_PORT,
        SPI_DEVICE,
        max_speed_hz=SPEED_HZ))

# Initialize display.
disp.begin()

def screen_image(img_path):

    # Load, Resize, and convert the image to RGB.
#    print(f'Loading image: {img_path}')
    img = Image.open(img_path)
    img = img.resize((WIDTH, HEIGHT)).convert('RGB')

    # Swap Red and Blue channels to correct BGR issue
    r, g, b = img.split()
    img = Image.merge("RGB", (b, g, r))
    disp.display(img)

