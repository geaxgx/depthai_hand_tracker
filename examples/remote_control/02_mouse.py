#!/usr/bin/env python3

README = """
Move your mouse pointer with your hand. 
The pointer moves when your hand is doing the ONE or TWO pose.
The difference between ONE and TWO is that in TWO the left button
is also pressed.

The mouse location is calculated from the index finger tip location.
An double exponential filter is used to limit jittering.

If you have multiple screens, you may have to modify the line:
monitor = get_monitors()[0]

"""

print(README)

from HandController import HandController

# Controlling the mouse
try:
    from pynput.mouse import Button, Controller
except ModuleNotFoundError:
    print("To run this demo, you need the python package: pynput")
    print("Can be installed with: pip install pynput")
    import sys
    sys.exit()
mouse = Controller()

# Get screen resolution 
try:
    from screeninfo import get_monitors
except ModuleNotFoundError:
    print("To run this demo, you need the python package: screeninfo")
    print("Can be installed with: pip install screeninfo")
    import sys
    sys.exit()
monitor = get_monitors()[0] # Replace '0' by the index of your screen in case of multiscreen
print(monitor)

# Smoothing filter
import numpy as np
class DoubleExpFilter:
    def __init__(self,smoothing=0.65,
                 correction=1.0,
                 prediction=0.85,
                 jitter_radius=250.,
                 max_deviation_radius=540.,
                 out_int=False):
        self.smoothing = smoothing
        self.correction = correction
        self.prediction = prediction
        self.jitter_radius = jitter_radius
        self.max_deviation_radius = max_deviation_radius
        self.count = 0
        self.filtered_pos = 0
        self.trend = 0
        self.raw_pos = 0
        self.out_int = out_int
        self.enable_scrollbars = False
    
    def reset(self):
        self.count = 0
        self.filtered_pos = 0
        self.trend = 0
        self.raw_pos = 0
    
    def update(self, pos):
        raw_pos = np.asanyarray(pos)
        if self.count > 0:
            prev_filtered_pos = self.filtered_pos
            prev_trend = self.trend
            prev_raw_pos = self.raw_pos
        if self.count == 0:
            self.shape = raw_pos.shape
            filtered_pos = raw_pos
            trend = np.zeros(self.shape)
            self.count = 1
        elif self.count == 1:
            filtered_pos = (raw_pos + prev_raw_pos)/2
            diff = filtered_pos - prev_filtered_pos
            trend = diff*self.correction + prev_trend*(1-self.correction)
            self.count = 2
        else:
            # First apply jitter filter
            diff = raw_pos - prev_filtered_pos
            length_diff = np.linalg.norm(diff)
            if length_diff <= self.jitter_radius:
                alpha = pow(length_diff/self.jitter_radius,1.5)
                # alpha = length_diff/self.jitter_radius
                filtered_pos = raw_pos*alpha \
                                + prev_filtered_pos*(1-alpha)
            else:
                filtered_pos = raw_pos
            # Now the double exponential smoothing filter
            filtered_pos = filtered_pos*(1-self.smoothing) \
                        + self.smoothing*(prev_filtered_pos+prev_trend)
            diff = filtered_pos - prev_filtered_pos
            trend = self.correction*diff + (1-self.correction)*prev_trend
        # Predict into the future to reduce the latency
        predicted_pos = filtered_pos + self.prediction*trend
        # Check that we are not too far away from raw data
        diff = predicted_pos - raw_pos
        length_diff = np.linalg.norm(diff)
        if length_diff > self.max_deviation_radius:
            predicted_pos = predicted_pos*self.max_deviation_radius/length_diff \
                        + raw_pos*(1-self.max_deviation_radius/length_diff)
        # Save the data for this frame
        self.raw_pos = raw_pos
        self.filtered_pos = filtered_pos
        self.trend = trend
        # Output the data
        if self.out_int:
            return predicted_pos.astype(int)
        else:
            return predicted_pos

smooth = DoubleExpFilter(smoothing=0.3, prediction=0.1, jitter_radius=700, out_int=True)

# Camera image size
cam_width = 1152
cam_height = 648


def move(event):
    # Use location of index
    x, y = event.hand.landmarks[8,:2]
    x /= cam_width
    x = 1 - x
    y /= cam_height
    e = 0.15
    p1 = monitor.width/(1-2*e)
    q1 = -p1*e
    mx = int(max(0, min(monitor.width-1, p1*x+q1)))
    et = 0.05
    eb= 0.4
    p2 = monitor.height/(1-et-eb)
    q2 = -p2*et
    my = int(max(0, min(monitor.height-1, p2*y+q2)))
    mx,my = smooth.update((mx,my))
    mouse.position = (mx+monitor.x, my+monitor.y)

def press_release(event):
    if event.trigger == "enter": 
        mouse.press(Button.left)
    elif event.trigger == "leave":
        mouse.release(Button.left)

def click(event):
    mouse.press(Button.left)
    mouse.release(Button.left)

config = {
    'renderer' : {'enable': True},
    
    'pose_actions' : [

        {'name': 'MOVE', 'pose':['ONE','TWO'], 'callback': 'move', "trigger":"continuous", "first_trigger_delay":0.1,},
        {'name': 'CLICK', 'pose':'TWO', 'callback': 'press_release', "trigger":"enter_leave", "first_trigger_delay":0.1},
    ]
}

HandController(config).loop()