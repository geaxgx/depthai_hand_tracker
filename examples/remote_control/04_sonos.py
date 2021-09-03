#!/usr/bin/env python3

README = """
Play/pause a Sonos player with the FIST or OK pose
Use the ONE pose to query song information (artist, title, album)
The sound volume can be changed by rotating an open hand (FIVE)

"""

print(README)

from HandController import HandController

# Controlling the Sonos player
try:
    import soco
except ModuleNotFoundError:
    print("To run this demo, you need the python package: soco")
    print("Can be installed with: pip install soco")
    import sys
    sys.exit()
sonos = soco.discovery.any_soco()

# Callbacks
def on_off(event):
    event.print_line()
    sonos_state = sonos.get_current_transport_info()['current_transport_state']
    if sonos_state == 'PLAYING':
        sonos.pause()
    else: # STOPPED or PAUSED_PLAYBACK
        sonos.play()

def song_info(event):
    event.print_line()
    track_info = sonos.get_current_track_info()
    track_info = f"{track_info['artist']}, {track_info['title']}, {track_info['album']}"
    print(track_info)


def change_volume(event):
    event.print_line()
    rotation = event.hand.rotation
    if rotation < -0.2:
        level = "+"
    elif rotation > 0.4:
        level = "-"
    else:
        return  
    volume = sonos.volume
    print("Volume:", volume)
    if level == "+":
        sonos.volume = min(100, volume + 1)
    else:
        sonos.volume = max(0, volume - 1)

config = {

    'tracker': {'args': {'body_pre_focusing': 'higher'}},

    'renderer' : {'enable': True},
    
    'pose_actions' : [
        {'name': 'ON_OFF', 'pose': ['FIST','OK'], 'callback': 'on_off'},
        {'name': 'VOLUME', 'pose': 'FIVE', 'callback': 'change_volume',
        "trigger":"periodic", "first_trigger_delay":0.3, "next_trigger_delay":0.3, },
        {'name': 'INFO', 'pose': 'ONE', 'callback': 'song_info'},
    ]
}

HandController(config).loop()