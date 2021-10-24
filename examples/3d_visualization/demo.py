#!/usr/bin/env python3

import sys
sys.path.append("../..")
from HandTrackerRenderer import HandTrackerRenderer
from HandTracker3DRenderer import HandTracker3DRenderer
from Filters import LandmarksSmoothingFilter
import argparse
import numpy as np
import cv2
from o3d_utils import Visu3D

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

class HandTracker3DRenderer:
    def __init__(self, tracker, smoothing=True):
        self.vis3d = Visu3D(zoom=0.7, segment_radius=10)
        z = min(tracker.img_h, tracker.img_w)/3
        self.vis3d.create_grid([0,tracker.img_h,-z],[tracker.img_w,tracker.img_h,-z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Floor
        self.vis3d.create_grid([0,0,z],[tracker.img_w,0,z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Wall
        self.vis3d.init_view()

        self.smoothing = smoothing
        self.filter = None
        if self.smoothing:
            if tracker.solo:
                self.filter = [LandmarksSmoothingFilter(
                    min_cutoff=0.01,
                    beta=40,
                    derivate_cutoff=1
                )]
            else:
                self.filter = [
                    LandmarksSmoothingFilter(min_cutoff=0.01,beta=40,derivate_cutoff=1),
                    LandmarksSmoothingFilter(min_cutoff=0.01,beta=40,derivate_cutoff=1)
                    ]

        self.nb_hands_in_previous_frame = 0

    def draw_hand(self, hand, i):
        # Denormalize z-component of 'norm_landmarks'
        lm_z = (hand.norm_landmarks[:,2:3] * hand.rect_w_a  / 0.4).astype(np.int)
        # ... and concatenates with x and y components of 'landmarks'
        points = np.hstack((hand.landmarks, lm_z))
        if self.smoothing:
            points = self.filter[i].apply(points, object_scale=hand.rect_w_a)

        radius = hand.rect_w_a / 30 # Thickness of segments depends on the hand size
        for i,a_b in enumerate(LINES_HAND):
            a, b = a_b
            self.vis3d.add_segment(points[a], points[b], radius=radius, color=[1*(1-hand.handedness),hand.handedness,0]) # if hand.handedness<0.5 else [0,1,0])
                    
    def draw(self, hands):
        if self.smoothing and len(hands) != self.nb_hands_in_previous_frame:
            for f in self.filter: f.reset()
        self.vis3d.clear()
        self.vis3d.try_move()
        self.vis3d.add_geometries()
        for i, hand in enumerate(hands):
            self.draw_hand(hand, i)
        self.vis3d.render()
        self.nb_hands_in_previous_frame = len(hands)

parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--edge', action="store_true",
#                     help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Path to a blob file for landmark model")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Detect one hand max")         
parser_tracker.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter (smoothing works only in solo mode)")           
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")    
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=30,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")                
args = parser.parse_args()

args.edge = True
if args.edge:
    from HandTrackerEdge import HandTracker
else:
    from HandTracker import HandTracker

dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

tracker = HandTracker(
        input_src=args.input, 
        solo=args.solo,
        stats=True,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=1,
        **tracker_args
        )

renderer3d = HandTracker3DRenderer(tracker, smoothing=not args.no_smoothing)
renderer2d = HandTrackerRenderer(tracker)

pause = False
hands = []

while True:
    # Run hand tracker on next frame
    if not pause:
        frame, hands, bag = tracker.next_frame()
        if frame is None: break
        # Render 2d frame
        frame = renderer2d.draw(frame, hands, bag)
        cv2.imshow("HandTracker", frame)
    key = cv2.waitKey(1)
    # Draw hands on open3d canvas
    renderer3d.draw(hands)
    if key == 27 or key == ord('q'):
        break
    elif key == 32: # space
        pause = not pause
    elif key == ord('s'):
        if renderer3d.filter:
            renderer3d.smoothing = not renderer3d.smoothing
renderer2d.exit()
tracker.exit()
