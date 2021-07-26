import cv2
import numpy as np

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

class HandTrackerRenderer:
    def __init__(self, 
                tracker,
                output=None):

        self.tracker = tracker

        # Rendering flags
        if self.tracker.use_lm:
            self.show_pd_box = False
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_handedness = False
            self.show_landmarks = True
            self.show_scores = False
            self.show_gesture = self.tracker.use_gesture
        else:
            self.show_pd_box = True
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_scores = False

        self.show_xyz_zone = self.show_xyz = self.tracker.xyz
        self.show_fps = True

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,self.tracker.video_fps,(self.tracker.img_w, self.tracker.img_h)) 

    def norm2abs(self, x_y):
        x = int(x_y[0] * self.tracker.frame_size - self.tracker.pad_w)
        y = int(x_y[1] * self.tracker.frame_size - self.tracker.pad_h)
        return [x, y]

    def draw_hand(self, hand):

        if self.tracker.use_lm:
            # (info_ref_x, info_ref_y): coords in the image of a reference point 
            # relatively to which hands information (score, handedness, xyz,...) are drawn
            # info_ref_x = np.min(hand.landmarks[:,0])
            # info_ref_y = np.max(hand.landmarks[:,1])
            info_ref_x = hand.landmarks[0,0]
            info_ref_y = np.max(hand.landmarks[:,1])

            # thick_coef is used to adapt the size of the draw landmarks features according to the size of the hand.
            thick_coef = hand.rect_w_a / 400

            if hand.lm_score > self.tracker.lm_score_thresh:
                if self.show_rot_rect:
                    cv2.polylines(self.frame, [np.array(hand.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
                if self.show_landmarks:
                    lines = [np.array([hand.landmarks[point] for point in line]).astype(np.int) for line in LINES_HAND]
                    cv2.polylines(self.frame, lines, False, (255, 0, 0), int(2+thick_coef), cv2.LINE_AA)
                    radius = int(2+thick_coef*4)
                    if self.tracker.use_gesture:
                        # color depending on finger state (1=open, 0=close, -1=unknown)
                        color = { 1: (0,255,0), 0: (0,0,255), -1:(0,255,255)}
                        cv2.circle(self.frame, (hand.landmarks[0][0], hand.landmarks[0][1]), radius, color[-1], -1)
                        for i in range(1,5):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.thumb_state], -1)
                        for i in range(5,9):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.index_state], -1)
                        for i in range(9,13):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.middle_state], -1)
                        for i in range(13,17):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.ring_state], -1)
                        for i in range(17,21):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.little_state], -1)
                    else:
                        for x,y in hand.landmarks[:,:2]:
                            cv2.circle(self.frame, (int(x), int(y)), radius, (0,128,255), -1)

                if self.show_handedness:
                    cv2.putText(self.frame, f"{hand.label.upper()} {hand.handedness:.2f}", 
                            (info_ref_x-90, info_ref_y+40), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) if hand.handedness > 0.5 else (0,0,255), 2)
                if self.show_scores:
                    cv2.putText(self.frame, f"Landmark score: {hand.lm_score:.2f}", 
                            (info_ref_x-90, info_ref_y+110), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
                if self.tracker.use_gesture and self.show_gesture:
                    cv2.putText(self.frame, hand.gesture, (info_ref_x-20, info_ref_y-50), 
                            cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

        if hand.pd_box is not None:
            box = hand.pd_box
            box_tl = self.norm2abs((box[0], box[1]))
            box_br = self.norm2abs((box[0]+box[2], box[1]+box[3]))
            if self.show_pd_box:
                cv2.rectangle(self.frame, box_tl, box_br, (0,255,0), 2)
            if self.show_pd_kps:
                for i,kp in enumerate(hand.pd_kps):
                    x_y = self.norm2abs(kp)
                    cv2.circle(self.frame, x_y, 6, (0,0,255), -1)
                    cv2.putText(self.frame, str(i), (x_y[0], x_y[1]+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                if self.tracker.use_lm:
                    x, y = info_ref_x - 90, info_ref_y + 80
                else:
                    x, y = box_tl[0], box_br[1]+60
                cv2.putText(self.frame, f"Palm score: {hand.pd_score:.2f}", 
                        (x, y), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)
            
        if self.show_xyz:
            if self.tracker.use_lm:
                x0, y0 = info_ref_x - 40, info_ref_y + 40
            else:
                x0, y0 = box_tl[0], box_br[1]+20
            cv2.rectangle(self.frame, (x0,y0), (x0+100, y0+85), (220,220,240), -1)
            cv2.putText(self.frame, f"X:{hand.xyz[0]/10:3.0f} cm", (x0+10, y0+20), cv2.FONT_HERSHEY_PLAIN, 1, (20,180,0), 2)
            cv2.putText(self.frame, f"Y:{hand.xyz[1]/10:3.0f} cm", (x0+10, y0+45), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
            cv2.putText(self.frame, f"Z:{hand.xyz[2]/10:3.0f} cm", (x0+10, y0+70), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
        if self.show_xyz_zone:
            # Show zone on which the spatial data were calculated
            cv2.rectangle(self.frame, tuple(hand.xyz_zone[0:2]), tuple(hand.xyz_zone[2:4]), (180,0,180), 2)


    def draw(self, frame, hands):
        self.frame_source = frame.copy() # Used for snapshot when debugging
        self.frame = frame
        for hand in hands:
            self.draw_hand(hand)
        return self.frame

    def exit(self):
        if self.output:
            self.output.release()

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
