import mrcnn
import numpy as np
import cv2
import sys
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import createConsoleLogger, setGlobalLogger
from pylibfreenect2 import LoggerLevel

# image scale hyper paramter
IMAGE_SCALE = 0.5

# select packet pipeline
try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()
print("Packet pipeline:", type(pipeline).__name__)

# Create and set logger
logger = createConsoleLogger(LoggerLevel.Debug)
setGlobalLogger(logger)

# check connected device
fn = Freenect2()
num_devices = fn.enumerateDevices()
if num_devices == 0:
    print("No device connected!")
    sys.exit(1) # exit program when no device connected

# get device
serial = fn.getDeviceSerialNumber(0)
device = fn.openDevice(serial, pipeline=pipeline)

# register listener
listener = SyncMultiFrameListener(FrameType.Color)
device.setColorFrameListener(listener)

# start device
device.start()

# main loop
while True:
    # get frame
    frames = listener.waitForNewFrame()

    # set input color image
    width, height = int(1920 * IMAGE_SCALE), int(1080 * IMAGE_SCALE)
    input = cv2.resize(frames["color"].asarray(), (width, height))[:, :, :-1]

    # run Mask-RCNN
    output = mrcnn.run(input)

    # visualize
    cv2.imshow("Input", input)
    cv2.imshow("Output", output)

    # release frame
    listener.release(frames)

    # update
    key = cv2.waitKey(delay=1)
    if key == ord('q') or key == 27:
        break # exit loop when 'q' or 'ESC' key was pressed

# stop and close device
device.stop()
device.close()

# exit program
sys.exit(0)
