import time, board, busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
pca = PCA9685(busio.I2C(board.SCL, board.SDA), address=0x40)
pca.frequency = 50
for ch in range(16):
    s = servo.Servo(pca.channels[ch], min_pulse=500, max_pulse=2500)
    print(f"channel {ch:2d} ...", flush=True)
    for angle in (70, 110, 70, 110, 90):
        s.angle = angle
        time.sleep(0.4)
    pca.channels[ch].duty_cycle = 0   # stop driving this channel
pca.deinit()
print("done")
