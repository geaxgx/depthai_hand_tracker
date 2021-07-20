import cv2
import numpy as np
from o3d_utils import Visu3D

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [1,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

class HandTracker3DRenderer:
    def __init__(self, tracker):

        self.tracker = tracker

        # Rendering flags
        self.show_fps = True


        self.vis3d = Visu3D(zoom=0.7, segment_radius=10)
        z = min(tracker.img_h, tracker.img_w)/3
        self.vis3d.create_grid([0,tracker.img_h,-z],[tracker.img_w,tracker.img_h,-z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Floor
        self.vis3d.create_grid([0,0,z],[tracker.img_w,0,z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Wall
        self.vis3d.init_view()


    def draw_hand(self, hand):

        lm_z = (hand.norm_landmarks[:,2:3] * hand.rect_w_a  / 0.4).astype(np.int)
        points = np.hstack((hand.landmarks, lm_z))
    
        lines = LINES_HAND
        radius = hand.rect_w_a / 33
        for i,a_b in enumerate(lines):
            a, b = a_b
            self.vis3d.add_segment(points[a], points[b], radius=radius, color=[1*(1-hand.handedness),hand.handedness,0]) # if hand.handedness<0.5 else [0,1,0])
                    

    def draw(self, frame, hands):

        self.vis3d.clear()
        self.vis3d.rotate()
        self.vis3d.add_geometries()
        for hand in hands:
            self.draw_hand(hand)
        self.vis3d.render()
        return self.frame


    def waitKey(self, delay=1):
        if self.show_fps:
                self.tracker.fps.draw(self.frame, orig=(50,50), size=1, color=(240,180,100))
        cv2.imshow("Hand tracking", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 
        if key == 32:
            # Pause on space bar
            key = cv2.waitKey(0)
            if key == ord('s'):
                print("Snapshot saved in snapshot.jpg")
                cv2.imwrite("snapshot.jpg", self.frame)
                cv2.imwrite("snapshot_src.jpg", self.frame_source)
        elif key == ord('1'):
            self.show_pd_box = not self.show_pd_box
        elif key == ord('2'):
            self.show_pd_kps = not self.show_pd_kps
        elif key == ord('3'):
            self.show_rot_rect = not self.show_rot_rect
        elif key == ord('4') and self.tracker.use_lm:
            self.show_landmarks = not self.show_landmarks
        elif key == ord('5') and self.tracker.use_lm:
            self.show_handedness = not self.show_handedness
        elif key == ord('6'):
            self.show_scores = not self.show_scores
        elif key == ord('7') and self.tracker.use_lm:
            if self.tracker.use_gesture:
                self.show_gesture = not self.show_gesture
        elif key == ord('8'):
            if self.tracker.xyz:
                self.show_xyz = not self.show_xyz    
        elif key == ord('9'):
            if self.tracker.xyz:
                self.show_xyz_zone = not self.show_xyz_zone 
        elif key == ord('f'):
            self.show_fps = not self.show_fps
        return key
