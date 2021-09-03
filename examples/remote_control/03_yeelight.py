#!/usr/bin/env python3

README = """
Switching on/off an IP bulb (brand Yeelight) with the FIST or OK pose
One color preset among 4 can be selected with ONE, TWO, THREE or FOUR pose
The brightness can be changed by rotating an open hand (FIVE)

"""

print(README)

from time import time
from HandController import HandController

# Controlling the bulb
try:
    from yeelight import Bulb, discover_bulbs
except ModuleNotFoundError:
    print("To run this demo, you need the python package: yeelight")
    print("Can be installed with: pip install yeelight")
    import sys
    sys.exit()
bulbs = discover_bulbs()
if bulbs == []:
    print("No Yeelight bulb detected !")
    import sys
    sys.exit()
bulb = Bulb(bulbs[0]['ip'])
bulb_last_alert_time = 0

# Callbacks
def toggle_light(event):
    event.print_line()
    bulb.toggle()

def change_preset(event):
    event.print_line()
    preset = event.name
    if preset == "PRESET 1":
        rgb = (255,255,255)
        bulb.set_rgb(*rgb)
    elif preset == "PRESET 2":
        rgb = (255,0,0)   
        bulb.set_rgb(*rgb)         
    elif preset == "PRESET 3":
        rgb = (0,0,255)
        bulb.set_rgb(*rgb)
    elif preset == "PRESET 4":
        rgb = (0,255,0)
        bulb.set_rgb(*rgb)


def change_brightness(event):
    event.print_line()
    rotation = event.hand.rotation
    if rotation < -0.2:
        level = "+"
    elif rotation > 0.4:
        level = "-"
    else:
        return  
    global bulb_last_alert_time
    brightness = int(bulb.get_properties()['bright'])
    if (brightness == 1 and level == "-") or (brightness == 100 and level == "+"):
        return
    if level == "+":
        bulb.set_brightness(min(100, brightness + 20))
    else:
        bulb.set_brightness(max(0, brightness - 20))

config = {

    'tracker': {'args': {'body_pre_focusing': 'higher'}},

    'renderer' : {'enable': True, 'args':{'output':'toggle_light.mp4'}},
    
    'pose_actions' : [
        {'name': 'ON_OFF', 'pose': 'FIST', 'callback': 'toggle_light'},
        {'name': 'PRESET 1', 'pose':'ONE', 'callback': 'change_preset'},
        {'name': 'PRESET 2', 'pose':'TWO', 'callback': 'change_preset'},
        {'name': 'PRESET 3', 'pose':'THREE', 'callback': 'change_preset'},
        {'name': 'PRESET 4', 'pose':'FOUR', 'callback': 'change_preset'},
        {'name': 'BRIGHTNESS', 'pose': 'FIVE', 'callback': 'change_brightness',
        "trigger":"periodic", "first_trigger_delay":0.3, "next_trigger_delay":0.3, },
    ]
}

HandController(config).loop()