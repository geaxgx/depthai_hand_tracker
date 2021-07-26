#!/usr/bin/env python3

README = """
This basic demo demonstrates the control the user have on how and when events are triggered,
by simply printing to the console event information each time an event is triggered. 

This control is defined with the following parameters of a pose-action:
- trigger : possible values: 
    - enter (default): an event is triggered once, when the pose begins,
    - enter_leave : two events are triggered, one when the pose begins and one when the pose ends,
    - periodic : events are triggered periodically as long as the pose stands.
                 The period is given by the parameter 'next_trigger_delay' in s.
    - continuous : events are triggered on every frame.

- first_trigger_delay: because false positive happen in pose recognition, 
you don't necessarily want to trigger an event on the first frame where the pose is recognized.
The 'first_trigger_delay' in seconds specifies how long the pose has to stand before triggering
an initial event.

"""

print(README)

from HandController import HandController

def trace(event):
    event.print_line()

def trace_rotation(event):
    event.print_line() 
    print("Rotation:", event.hand.rotation) 

def trace_index_finger_tip(event):
    event.print_line() 
    x, y = event.hand.landmarks[8,:2]
    print(f"Index finger tip : x={x}  y={y}") 

config = {
    'renderer' : {'enable': True},
    
    'pose_actions' : [
        {'name': '1_right_enter', 'pose':'ONE', 'hand':'right', 'callback': 'trace',"trigger":"enter", "first_trigger_delay":0.3},
        {'name': '2_right_enter_leave', 'pose':['TWO','PEACE'], 'hand':'right', 'callback': 'trace',"trigger":"enter_leave"},
        {'name': '3_right_periodic_1s', 'pose':'THREE', 'hand':'right', 'callback': 'trace', "trigger":"periodic", "first_trigger_delay":0, "next_trigger_delay": 1},
        {'name': '4_right_periodic_0.3s', 'pose':'FOUR', 'hand':'right', 'callback': 'trace', "trigger":"periodic", "first_trigger_delay":0, "next_trigger_delay": 0.3},
        {'name': '5_periodic_rotation', 'pose':'FIVE', 'callback': 'trace_rotation', "trigger":"periodic", "first_trigger_delay":0, "next_trigger_delay": 0.2},
        {'name': '1_left_continuous_xy', 'pose':'ONE', 'hand':'left', 'callback': 'trace_index_finger_tip',"trigger":"continuous"},
    ]
}

HandController(config).loop()