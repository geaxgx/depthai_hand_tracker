import sys
sys.path.append("../..")
import datetime
from time import monotonic

ALL_POSES = ["ONE","TWO","THREE","FOUR","FIVE","FIST","PEACE","OK"]

# Default values for config parameters
# Each one of these parameters can be superseded by a new value if specified in client code
DEFAULT_CONFIG = {
    'pose_params': 
    {
        "callback": "_DEFAULT_",
        "hand": "any",
        "trigger": "enter", 
        "first_trigger_delay": 0.3, 
        "next_trigger_delay": 0.3, 
        "max_missing_frames": 3,
    },

    'tracker': 
    { 
        'version': 'edge',

        'args': 
        {
            'pd_score_thresh': 0.6,
            'pd_nms_thresh': 0.3,
            'lm_score_thresh': 0.5, 
            'solo': True,
            'internal_fps': 30,
            'internal_frame_height': 640,
            'use_gesture': True
        },
    },

    'renderer':
    {   
        'enable': False,

        'args':
        {
            'output': None,
        }

    }
}

class Event:
    def __init__(self, category, hand, pose_action, trigger):
        self.category = category
        self.hand = hand
        if hand:
            self.handedness = hand.label
            self.pose = hand.gesture
        else:
            self.handedness = None
            self.pose = None
        self.name = pose_action["name"]
        self.callback = pose_action["callback"]
        self.trigger = trigger
        self.time = datetime.datetime.now()
    def print(self):
        attrs = vars(self)
        print("--- EVENT :")
        print('\n'.join("\t%s: %s" % item for item in attrs.items()))
    def print_line(self):
        print(f"{self.time.strftime('%H:%M:%S.%f')[:-3]} : {self.category} {self.name} [{self.pose}] - hand: {self.handedness} - trigger: {self.trigger} - callback: {self.callback}")
        


class PoseEvent(Event):
    def __init__(self, hand, pose_action, trigger):
        super().__init__("Pose",
                    hand,
                    pose_action,
                    trigger = trigger)

class EventHist:
    def __init__(self, triggered=False, first_triggered=False, time=0, frame_nb=0):
        self.triggered = triggered
        self.first_triggered = first_triggered
        self.time = time
        self.frame_nb = frame_nb

def default_callback(event):
    event.print_line()

def merge_dicts(d1, d2):
    """
    Merge 2 dictionaries. The 2nd dictionary's values overwrites those from the first
    """
    return {**d1, **d2}

def merge_config(c1, c2):
    """
    Merge 2 configs c1 and c2 (where c1 is the default config and c2 the user defined config).
    A config is a python dictionary. The result config takes key:value from c1 if key
    is not present in c2, and key:value from c2 otherwise (key is either present both in 
    in c1 and c2, or only in c2). 
    Note that merge_config is recursive : value can itself be a dictionary.
    """
    res = {}
    for k1,v1 in c1.items():
        if k1 in c2:
            if isinstance(v1, dict):
                assert isinstance(c2[k1], dict), f"{c2[k1]} should be a dictionary"
                res[k1] = merge_config(v1, c2[k1])
            else:
                res[k1] = c2[k1]
        else:
            res[k1] = v1
    for k2,v2 in c2.items():
        if k2 not in c1:
            res[k2] = v2
    return res

def check_mandatory_keys(dic, mandatory_keys):
    """
    Check that mandatory keys are present in a dic
    """
    for k in mandatory_keys:
        assert k in dic.keys(), f"Mandatory key '{k}' not present in {dic}"

class HandController:
    def __init__(self, config={}):
        self.config = merge_config(DEFAULT_CONFIG, config)

        # HandController will run callback functions defined in the calling app
        # self.caller_globals contains the globals from the calling app (including callbacks)
        self.caller_globals = sys._getframe(1).f_globals # Or vars(sys.modules['__main__'])

        # Parse pose config
        # Pose list is stored in self.poses
        self.parse_poses()

        # Keep records of previous pose status 
        self.poses_hist = [EventHist() for i in range(len(self.pose_actions))]

        # HandTracker
        tracker_version = self.config['tracker']['version']
        if tracker_version == 'edge':
            from HandTrackerEdge import HandTracker
        else: # 'host'
            from HandTracker import HandTracker
        # Forcing solo mode and use_gesture
        self.config['tracker']['args']['solo'] = True
        self.config['tracker']['args']['use_gesture'] = True
        # Init tracker
        self.tracker = HandTracker(**self.config['tracker']['args'])

        # Renderer
        self.use_renderer = self.config['renderer']['enable']
        if self.use_renderer:
            from HandTrackerRenderer import HandTrackerRenderer
            self.renderer = HandTrackerRenderer(self.tracker, **self.config['renderer']['args'])

        self.frame_nb = 0
        

    def parse_poses(self):
        """
        The part of the config related to poses looks like: 
        'pose_params': {"trigger": "enter", 
                        "first_trigger_delay":0.6, 
                        "next_trigger_delay":0.6, 
                        "max_missing_frames":3},
    
        'pose_actions' : [
            {'name': 'LIGHT', 'pose':'ONE', 'hand':'left', 'callback': 'set_context'},
            {'name': 'TV', 'pose':'TWO', 'hand':'left', 'callback': 'set_context'},
            ]
        
        In the 'pose_actions' list, one element is a dict which have :
            - 2 mandatory key: 
                - name: arbitrary name chosen by the user,
                - pose : one or a list of poses (from the predefined poses listed in ALL_POSES)
                            or keyword 'ALL' to specify any pose from ALL_POSES 
            - optional keys which are the keys of DEFAULT_CONFIG['pose_params']:
                - hand: specify the handedness = hand used to make the pose.
                        Values: 'left', 'right', 'any' (default)
        """
        mandatory_keys = ['name', 'pose']
        optional_keys = self.config['pose_params'].keys()
        self.pose_actions = []
        if 'pose_actions' in self.config:
            for pa in self.config['pose_actions']:
                check_mandatory_keys(pa, mandatory_keys)
                pose = pa['pose']
                if isinstance(pose, list):
                    for x in pose:
                        assert x in ALL_POSES, f"Incorrect pose {x} in {pa} !"
                elif pose == 'ALL':
                    pa['pose'] = ALL_POSES
                else:
                    # 'pose' is a single pose. Transform it into a list
                    assert pose in ALL_POSES, f"Incorrect pose {pose} in {pa} !"
                    pa['pose'] = [pose]
                optional_args = {k:pa.get(k, self.config['pose_params'][k]) for k in optional_keys}
                mandatory_args = { k:pa[k] for k in mandatory_keys}
                all_args = merge_dicts(mandatory_args, optional_args)
                self.pose_actions.append(all_args)
            

    def generate_events(self, hands):

        events = []

        # We are in solo mode -> either hands=[] or hands=[hand]
        hand = hands[0] if hands else None

        for i, pa in enumerate(self.pose_actions):
            hist = self.poses_hist[i]
            trigger = pa['trigger']
            if hand and hand.gesture and \
                (hand.label == pa['hand'] or pa['hand'] == 'any') and \
                hand.gesture in pa['pose']:
                if trigger == "continuous":
                    events.append(PoseEvent(hand, pa, "continuous"))
                else: # trigger in ["enter", "enter_leave", "periodic"]:
                    if not hist.triggered:
                        if hist.time != 0 and (self.frame_nb - hist.frame_nb <= pa['max_missing_frames']):
                            if  hist.time and \
                                ((hist.first_triggered and self.now - hist.time > pa['next_trigger_delay']) or \
                                    (not hist.first_triggered and self.now - hist.time > pa['first_trigger_delay'])):
                                
                                if trigger == "enter" or trigger == "enter_leave":
                                    hist.triggered = True
                                    events.append(PoseEvent(hand, pa, "enter"))
                                else: # "periodic"
                                    hist.time = self.now
                                    hist.first_triggered = True
                                    events.append(PoseEvent(hand, pa, "periodic"))
                                
                        else:
                            hist.time = self.now
                            hist.first_triggered = False
                    else:
                        if self.frame_nb - hist.frame_nb > pa['max_missing_frames']:
                            hist.time = self.now
                            hist.triggered = False
                            hist.first_triggered = False
                            if trigger == "enter_leave":
                                events.append(PoseEvent(hand, pa, "leave"))
                hist.frame_nb = self.frame_nb

            else:
                if hist.triggered and self.frame_nb - hist.frame_nb > pa['max_missing_frames']:
                    hist.time = self.now
                    hist.triggered = False
                    hist.first_triggered = False 
                    if trigger == "enter_leave":
                        events.append(PoseEvent(hand, pa, "leave")) 
        return events    

    def process_events(self, events):
        for e in events:
            if e.callback == "_DEFAULT_":
                default_callback(e)
            else:
                self.caller_globals[e.callback](e)

    def loop(self):
        while True:
            self.now = monotonic()
            frame, hands, bag = self.tracker.next_frame()
            if frame is None: break
            self.frame_nb += 1
            events = self.generate_events(hands)
            self.process_events(events)

            if self.use_renderer:
                frame = self.renderer.draw(frame, hands, bag)
                key = self.renderer.waitKey(delay=1)
                if key == 27 or key == ord('q'):
                    break
        self.renderer.exit()
        self.tracker.exit()
            


