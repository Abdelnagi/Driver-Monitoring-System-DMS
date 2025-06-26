# Thresholds for yawning and blinking

# If the driver yawns continuously for at least 1 second, it triggers an alert and increments the yawn counter
yawning_threshold = 1.0

# If the driver's eyes remain closed for at least 3 seconds, it's considered a microsleep and triggers an alert
microsleep_threshold = 3.0

# Minimum Eye Aspect Ratio (EAR); below this means the eyes are considered closed
eye_open_threshold = 0.30

# Minimum Mouth Aspect Ratio (MAR); below this means the mouth is considered closed
mouth_closed_threshold = 0.50

# If the driver's head remains in an abnormal position (pitch/yaw/roll) for 20 seconds, it's assumed they’ve fallen asleep
abnormal_head_sleep_threshold = 20
