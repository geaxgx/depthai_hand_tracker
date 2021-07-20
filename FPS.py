"""
@author: geaxx
"""
import time
import cv2
from collections import deque

def now():
    return time.perf_counter()
    
class FPS: # To measure the number of frame per second
    def __init__(self, average_of=30):
        self.timestamps = deque(maxlen=average_of)
        self.nbf = -1
        
    def update(self):
        self.timestamps.append(time.monotonic())
        if len(self.timestamps) == 1:
            self.start = self.timestamps[0]
            self.fps = 0
        else:
            self.fps = (len(self.timestamps)-1)/(self.timestamps[-1]-self.timestamps[0])  
        self.nbf+=1
    
    def get(self):
        return self.fps
    
    def get_global(self):
        return self.nbf/(self.timestamps[-1] - self.start)
        
    def nb_frames(self):
        return self.nbf+1

    def draw(self, win, orig=(10,30), font=cv2.FONT_HERSHEY_SIMPLEX, size=2, color=(0,255,0), thickness=2):
        cv2.putText(win,f"FPS={self.get():.2f}",orig,font,size,color,thickness)

if __name__ == "__main__":
    fps = FPS()
    for i in range(50):
        fps.update()
        print(f"fps = {fps.get()}")
        time.sleep(0.1)
    global_fps, nb_frames = fps.get_global()
    print(f"Global fps : {global_fps} ({nb_frames} frames)")
