import numpy as np
from collections import namedtuple
import mediapipe_utils as mpu
import depthai as dai
import cv2
from pathlib import Path
from FPS import FPS, now
import time
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL = str(SCRIPT_DIR / "models/hand_landmark_sh4.blob")

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()



class HandTracker:
    """
    Mediapipe Hand Tracker for depthai
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host (Edge mode only),
                    - a file path of an image or a video,
                    - an integer (eg 0) for a webcam id,
    - pd_model: palm detection model blob file (if None, takes the default value PALM_DETECTION_MODEL),
    - pd_nms_thresh: NMS threshold,
    - pd_score: confidence score to determine whether a detection is reliable (a float between 0 and 1).
    - use_lm: boolean. When True, run landmark model. Otherwise, only palm detection model is run
    - lm_model: landmark model blob file
                    - None : the default blob file LANDMARK_MODEL,
                    - a path of a blob file. 
    - lm_score_thresh : confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1).
    - solo: boolean, when True detect one hand max (much faster since we run the pose detection model only if no hand was detected in the previous frame)
    - xyz : boolean, when True get the (x, y, z) coords of the detected hands (if the device supports depth measure).
    - crop : boolean which indicates if square cropping on source images is applied or not
    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - resolution : sensor resolution "full" (1920x1080) or "ultra" (3840x2160),
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                    The width is calculated accordingly to height and depends on value of 'crop'
    - use_gesture : boolean, when True, recognize hand poses froma predefined set of poses
                    (ONE, TWO, THREE, FOUR, FIVE, OK, PEACE, FIST)
    - stats : boolean, when True, display some statistics when exiting.   
    - trace: boolean, when True print some debug messages (only in Edge mode)   
    """
    def __init__(self, input_src=None,
                pd_model=PALM_DETECTION_MODEL, 
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                use_lm=True,
                lm_model=LANDMARK_MODEL,
                lm_score_thresh=0.5,
                solo=False,
                xyz=False,
                crop=False,
                internal_fps=28,
                resolution="full",
                internal_frame_height=640,
                use_gesture=False,
                stats=False,
                trace=False
                ):

        self.pd_model = pd_model
        print(f"Palm detection blob file : {self.pd_model}")
        if use_lm:
            self.lm_model = lm_model
            print(f"Landmark blob file : {self.lm_model}")
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_score_thresh = lm_score_thresh
        if not use_lm and solo:
            print("Warning: solo mode desactivated when not using landmarks")
            self.solo = False
        else:
            self.solo = solo
        self.xyz = False
        self.crop = crop 
        self.internal_fps = internal_fps     
        self.stats = stats
        self.use_gesture = use_gesture

        self.device = dai.Device()

        if input_src == None or input_src == "rgb" or input_src == "rgb_laconic":
            # Note that here (in Host mode), specifying "rgb_laconic" has no effect
            # Color camera frames are systematically transferred to the host
            self.input_type = "rgb" # OAK* internal color camera
            self.internal_fps = internal_fps 
            print(f"Internal camera FPS set to: {self.internal_fps}")
            if resolution == "full":
                self.resolution = (1920, 1080)
            elif resolution == "ultra":
                self.resolution = (3840, 2160)
            else:
                print(f"Error: {resolution} is not a valid resolution !")
                sys.exit()
            print("Sensor resolution:", self.resolution)

            if xyz:
                # Check if the device supports stereo
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                    self.xyz = True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")

            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps
            
            if self.crop:
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height, self.resolution)
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2
            else:
                width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * self.resolution[0] / self.resolution[1], self.resolution, is_height=False)
                self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0

            print(f"Internal camera image size: {self.img_w} x {self.img_h} - crop_w:{self.crop_w} pad_h: {self.pad_h}")

        elif input_src.endswith('.jpg') or input_src.endswith('.png') :
            self.input_type= "image"
            self.img = cv2.imread(input_src)
            self.video_fps = 25
            self.img_h, self.img_w = self.img.shape[:2]
        else:
            self.input_type = "video"
            if input_src.isdigit():
                input_type = "webcam"
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Video FPS:", self.video_fps)
        
        if self.input_type != "rgb":
            print(f"Original frame size: {self.img_w}x{self.img_h}")
            if self.crop:
                self.frame_size = min(self.img_w, self.img_h) # // 16 * 16
            else:
                self.frame_size = max(self.img_w, self.img_h) #// 16 * 16
            self.crop_w = max((self.img_w - self.frame_size) // 2, 0)
            if self.crop_w: print("Cropping on width :", self.crop_w)
            self.crop_h = max((self.img_h - self.frame_size) // 2, 0)
            if self.crop_h: print("Cropping on height :", self.crop_h)

            self.pad_w = max((self.frame_size - self.img_w) // 2, 0)
            if self.pad_w: print("Padding on width :", self.pad_w)
            self.pad_h = max((self.frame_size - self.img_h) // 2, 0)
            if self.pad_h: print("Padding on height :", self.pad_h)
                     
            print(f"Frame working size: {self.img_w}x{self.img_h}")
        

        # Create SSD anchors 
        self.anchors = mpu.generate_handtracker_anchors()
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Define and start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues 
        if self.input_type == "rgb":
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
            if self.solo:
                self.q_pre_pd_manip_cfg = self.device.getInputQueue(name="pre_pd_manip_cfg")
            if self.use_lm:
                self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
                self.q_lm_in = self.device.getInputQueue(name="lm_in")
            if self.xyz:
                self.q_spatial_data = self.device.getOutputQueue(name="spatial_data_out", maxSize=4, blocking=False)
                self.q_spatial_config = self.device.getInputQueue("spatial_calc_config_in")

        else:
            self.q_pd_in = self.device.getInputQueue(name="pd_in")
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
            if self.use_lm:
                self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
                self.q_lm_in = self.device.getInputQueue(name="lm_in")

        self.fps = FPS()

        self.nb_pd_inferences = 0
        self.nb_lm_inferences = 0
        self.nb_spatial_requests = 0
        self.glob_pd_rtrip_time = 0
        self.glob_lm_rtrip_time = 0
        self.glob_spatial_rtrip_time = 0

        if self.solo:
            self.use_previous_landmarks = False

        
    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)
        self.pd_input_length = 128

        if self.input_type == "rgb":
            # ColorCamera
            print("Creating Color Camera...")
            cam = pipeline.createColorCamera()
            if self.resolution[0] == 1920:
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            else:
                cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam.setInterleaved(False)
            cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
            cam.setFps(self.internal_fps)
            if self.solo:
                # In solo mode, we need to control the flow of frames
                # going to the palm detection model.
                # The control is made from the host via the inputConfig port of 
                # ImageManip node
                pre_pd_manip = pipeline.createImageManip()
                pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
                pre_pd_manip.setWaitForConfigInput(True)
                pre_pd_manip.inputImage.setQueueSize(1)
                pre_pd_manip.inputImage.setBlocking(False)
                cam.preview.link(pre_pd_manip.inputImage)
                if self.crop:
                    cam.setVideoSize(self.frame_size, self.frame_size)
                    # cam.setPreviewSize(self.frame_size, self.frame_size)
                    cam.setPreviewSize(self.pd_input_length, self.pd_input_length)
                else: 
                    cam.setVideoSize(self.img_w, self.img_h)
                    cam.setPreviewSize(self.img_w, self.img_h)
                # cfg_pre_pd : the config that will be send by the host to the pre_pd_manip
                self.cfg_pre_pd = dai.ImageManipConfig()
                self.cfg_pre_pd.setResizeThumbnail(self.pd_input_length, self.pd_input_length)
                
                pre_pd_manip_cfg_in = pipeline.createXLinkIn()
                pre_pd_manip_cfg_in.setStreamName("pre_pd_manip_cfg")
                pre_pd_manip_cfg_in.out.link(pre_pd_manip.inputConfig) 

            else:
                # In multi mode, no need to control the flow, but we may still need
                # an ImageManip node for letterboxing (if crop is False)
                if self.crop:
                    cam.setVideoSize(self.frame_size, self.frame_size)
                    cam.setPreviewSize(self.pd_input_length, self.pd_input_length)
                else: 
                    cam.setVideoSize(self.img_w, self.img_h)
                    cam.setPreviewSize(self.img_w, self.img_h)
                    # Letterboxing
                    print("Creating letterboxing image manip...")
                    # pre_pd_manip = pipeline.create(dai.node.ImageManip)
                    pre_pd_manip = pipeline.createImageManip()
                    pre_pd_manip.initialConfig.setResizeThumbnail(self.pd_input_length, self.pd_input_length)
                    pre_pd_manip.setMaxOutputFrameSize(3*self.pd_input_length**2)
                    pre_pd_manip.inputImage.setQueueSize(1)
                    pre_pd_manip.inputImage.setBlocking(False)
                    cam.preview.link(pre_pd_manip.inputImage)

            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            cam_out.input.setQueueSize(1)
            cam_out.input.setBlocking(False)
            cam.video.link(cam_out.input)

            if self.xyz:
                print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes...")
                # For now, RGB needs fixed focus to properly align with depth.
                # This value was used during calibration
                cam.initialControl.setManualFocus(130)

                mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
                left = pipeline.createMonoCamera()
                left.setBoardSocket(dai.CameraBoardSocket.LEFT)
                left.setResolution(mono_resolution)
                left.setFps(self.internal_fps)

                right = pipeline.createMonoCamera()
                right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
                right.setResolution(mono_resolution)
                right.setFps(self.internal_fps)

                stereo = pipeline.createStereoDepth()
                stereo.setConfidenceThreshold(230)
                # LR-check is required for depth alignment
                stereo.setLeftRightCheck(True)
                stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
                stereo.setSubpixel(False)  # subpixel True -> latency

                spatial_location_calculator = pipeline.createSpatialLocationCalculator()
                spatial_location_calculator.setWaitForConfigInput(True)
                spatial_location_calculator.inputDepth.setBlocking(False)
                spatial_location_calculator.inputDepth.setQueueSize(1)

                spatial_data_out = pipeline.createXLinkOut()
                spatial_data_out.setStreamName("spatial_data_out")
                spatial_data_out.input.setQueueSize(1)
                spatial_data_out.input.setBlocking(False)

                spatial_calc_config_in = pipeline.createXLinkIn()
                spatial_calc_config_in.setStreamName("spatial_calc_config_in")

                left.out.link(stereo.left)
                right.out.link(stereo.right)    

                stereo.depth.link(spatial_location_calculator.inputDepth)

                spatial_location_calculator.out.link(spatial_data_out.input)
                spatial_calc_config_in.out.link(spatial_location_calculator.inputConfig)

        # Define palm detection model
        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(self.pd_model)
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)
        # Specify that network takes latest arriving frame in non-blocking manner
        # Palm detection input                 
        if self.input_type == "rgb":
            pd_nn.input.setQueueSize(1)
            pd_nn.input.setBlocking(False)
            if self.crop:
                cam.preview.link(pd_nn.input)
            else:
                pre_pd_manip.out.link(pd_nn.input)
        else:
            pd_in = pipeline.createXLinkIn()
            pd_in.setStreamName("pd_in")
            pd_in.out.link(pd_nn.input)
        # Palm detection output
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)
        

         # Define hand landmark model
        if self.use_lm:
            print("Creating Hand Landmark Neural Network...")          
            lm_nn = pipeline.createNeuralNetwork()
            lm_nn.setBlobPath(self.lm_model)
            lm_nn.setNumInferenceThreads(1)
            # Hand landmark input
            self.lm_input_length = 224
            lm_in = pipeline.createXLinkIn()
            lm_in.setStreamName("lm_in")
            lm_in.out.link(lm_nn.input)
            # Hand landmark output
            lm_out = pipeline.createXLinkOut()
            lm_out.setStreamName("lm_out")
            lm_nn.out.link(lm_out.input)
            
        print("Pipeline created.")
        return pipeline        
   
    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,18)) # 896x18
        # Decode bboxes
        self.hands = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=self.solo)
        # Non maximum suppression (not needed if solo)
        if not self.solo:
            self.hands = mpu.non_max_suppression(self.hands, self.pd_nms_thresh)
        if self.use_lm:
            mpu.detections_to_rect(self.hands)
            mpu.rect_transformation(self.hands, self.frame_size, self.frame_size)


    def lm_postprocess(self, hand, inference):
        hand.lm_score = inference.getLayerFp16("Identity_1")[0]  
        if hand.lm_score > self.lm_score_thresh:  
            hand.handedness = inference.getLayerFp16("Identity_2")[0]
            hand.label = "right" if hand.handedness > 0.5 else "left"
            lm_raw = np.array(inference.getLayerFp16("Identity_dense/BiasAdd/Add")).reshape(-1,3)
            # hand.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the square rotated body bounding box
            hand.norm_landmarks = lm_raw / self.lm_input_length
            # hand.norm_landmarks[:,2] /= 0.4

            # Now calculate hand.landmarks = the landmarks in the image coordinate system (in pixel)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in hand.rect_points[1:]], dtype=np.float32) # hand.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(hand.norm_landmarks[:,:2], axis=0)
            # lm_z = hand.norm_landmarks[:,2:3] * hand.rect_w_a  / 0.4
            hand.landmarks = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)

            if self.use_gesture: self.recognize_gesture(hand)
            # # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points

            # if self.pad_h > 0:
            #     hand.landmarks[:,1] -= self.pad_h
            #     for i in range(len(hand.rect_points)):
            #         hand.rect_points[i][1] -= self.pad_h
            # if self.pad_w > 0:
            #     hand.landmarks[:,0] -= self.pad_w
            #     for i in range(len(hand.rect_points)):
            #         hand.rect_points[i][0] -= self.pad_w

    def recognize_gesture(self, r):           

        # Finger states
        # state: -1=unknown, 0=close, 1=open
        d_3_5 = mpu.distance(r.norm_landmarks[3], r.norm_landmarks[5])
        d_2_3 = mpu.distance(r.norm_landmarks[2], r.norm_landmarks[3])
        angle0 = mpu.angle(r.norm_landmarks[0], r.norm_landmarks[1], r.norm_landmarks[2])
        angle1 = mpu.angle(r.norm_landmarks[1], r.norm_landmarks[2], r.norm_landmarks[3])
        angle2 = mpu.angle(r.norm_landmarks[2], r.norm_landmarks[3], r.norm_landmarks[4])
        r.thumb_angle = angle0+angle1+angle2
        if angle0+angle1+angle2 > 460 and d_3_5 / d_2_3 > 1.2: 
            r.thumb_state = 1
        else:
            r.thumb_state = 0

        if r.norm_landmarks[8][1] < r.norm_landmarks[7][1] < r.norm_landmarks[6][1]:
            r.index_state = 1
        elif r.norm_landmarks[6][1] < r.norm_landmarks[8][1]:
            r.index_state = 0
        else:
            r.index_state = -1

        if r.norm_landmarks[12][1] < r.norm_landmarks[11][1] < r.norm_landmarks[10][1]:
            r.middle_state = 1
        elif r.norm_landmarks[10][1] < r.norm_landmarks[12][1]:
            r.middle_state = 0
        else:
            r.middle_state = -1

        if r.norm_landmarks[16][1] < r.norm_landmarks[15][1] < r.norm_landmarks[14][1]:
            r.ring_state = 1
        elif r.norm_landmarks[14][1] < r.norm_landmarks[16][1]:
            r.ring_state = 0
        else:
            r.ring_state = -1

        if r.norm_landmarks[20][1] < r.norm_landmarks[19][1] < r.norm_landmarks[18][1]:
            r.little_state = 1
        elif r.norm_landmarks[18][1] < r.norm_landmarks[20][1]:
            r.little_state = 0
        else:
            r.little_state = -1

        # Gesture
        if r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 1 and r.little_state == 1:
            r.gesture = "FIVE"
        elif r.thumb_state == 0 and r.index_state == 0 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "FIST"
        elif r.thumb_state == 1 and r.index_state == 0 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "OK" 
        elif r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "PEACE"
        elif r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "ONE"
        elif r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 0 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "TWO"
        elif r.thumb_state == 1 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 0 and r.little_state == 0:
            r.gesture = "THREE"
        elif r.thumb_state == 0 and r.index_state == 1 and r.middle_state == 1 and r.ring_state == 1 and r.little_state == 1:
            r.gesture = "FOUR"
        else:
            r.gesture = None
            
    def spatial_loc_roi_from_palm_center(self, hand):
        half_size = int(hand.pd_box[2] * self.frame_size / 2)
        zone_size = max(half_size//2, 8)
        rect_center = dai.Point2f(int(hand.pd_box[0]*self.frame_size) + half_size - zone_size//2 + self.crop_w, int(hand.pd_box[1]*self.frame_size) + half_size - zone_size//2 - self.pad_h)
        rect_size = dai.Size2f(zone_size, zone_size)
        return dai.Rect(rect_center, rect_size)

    def spatial_loc_roi_from_wrist_landmark(self, hand):
        zone_size = max(int(hand.rect_w_a / 10), 8)
        rect_center = dai.Point2f(*(hand.landmarks[0]-np.array((zone_size//2 - self.crop_w, zone_size//2 + self.pad_h))))
        rect_size = dai.Size2f(zone_size, zone_size)
        return dai.Rect(rect_center, rect_size)

    def query_xyz(self, spatial_loc_roi_func):
        conf_datas = []
        for h in self.hands:
            conf_data = dai.SpatialLocationCalculatorConfigData()
            conf_data.depthThresholds.lowerThreshold = 100
            conf_data.depthThresholds.upperThreshold = 10000
            conf_data.roi = spatial_loc_roi_func(h)
            conf_datas.append(conf_data)
        if len(conf_datas) > 0:
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.setROIs(conf_datas)
            
            spatial_rtrip_time = now()
            self.q_spatial_config.send(cfg)

            # Receives spatial locations
            spatial_data = self.q_spatial_data.get().getSpatialLocations()
            self.glob_spatial_rtrip_time += now() - spatial_rtrip_time
            self.nb_spatial_requests += 1
            for i,sd in enumerate(spatial_data):
                self.hands[i].xyz_zone =  [
                    int(sd.config.roi.topLeft().x) - self.crop_w,
                    int(sd.config.roi.topLeft().y),
                    int(sd.config.roi.bottomRight().x) - self.crop_w,
                    int(sd.config.roi.bottomRight().y)
                    ]
                self.hands[i].xyz = [
                    sd.spatialCoordinates.x,
                    sd.spatialCoordinates.y,
                    sd.spatialCoordinates.z
                    ]

    def next_frame(self):

        self.fps.update()
        if self.input_type == "rgb":
            if self.solo and not self.use_previous_landmarks:
                self.q_pre_pd_manip_cfg.send(self.cfg_pre_pd)
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()
            if self.pad_h:
                square_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
            else:
                square_frame = video_frame
        else:
            if self.input_type == "image":
                frame = self.img.copy()
            else:
                ok, frame = self.cap.read()
                if not ok:
                    return None, None
            # Cropping and/or padding of the video frame
            video_frame = frame[self.crop_h:self.crop_h+self.frame_size, self.crop_w:self.crop_w+self.frame_size]
            if self.pad_h or self.pad_w:
                square_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
            else:
                square_frame = video_frame

            if not self.solo or not self.use_previous_landmarks:
                frame_nn = dai.ImgFrame()
                frame_nn.setTimestamp(time.monotonic())
                frame_nn.setWidth(self.pd_input_length)
                frame_nn.setHeight(self.pd_input_length)
                frame_nn.setData(to_planar(square_frame, (self.pd_input_length, self.pd_input_length)))
                self.q_pd_in.send(frame_nn)
                pd_rtrip_time = now()

        # Get palm detection
        if not self.solo or not self.use_previous_landmarks:
            inference = self.q_pd_out.get()
            if self.input_type != "rgb": 
                self.glob_pd_rtrip_time += now() - pd_rtrip_time
            self.pd_postprocess(inference)
            self.nb_pd_inferences += 1   
        else:
            self.hands = [self.hand_from_landmarks]

        # Hand landmarks, send requests
        if self.use_lm:
            for i,h in enumerate(self.hands):
                img_hand = mpu.warp_rect_img(h.rect_points, square_frame, self.lm_input_length, self.lm_input_length)
                nn_data = dai.NNData()   
                nn_data.setLayer("input_1", to_planar(img_hand, (self.lm_input_length, self.lm_input_length)))
                self.q_lm_in.send(nn_data)
                if i == 0: lm_rtrip_time = now() # We measure only for the first hand

            for i,h in enumerate(self.hands):
                inference = self.q_lm_out.get()
                if i == 0: self.glob_lm_rtrip_time += now() - lm_rtrip_time
                self.lm_postprocess(h, inference)
                self.nb_lm_inferences += 1
            self.hands = [ h for h in self.hands if h.lm_score > self.lm_score_thresh]

            if self.xyz:
                self.query_xyz(self.spatial_loc_roi_from_wrist_landmark)

            if self.solo:
                if len(self.hands):
                    # hand_from_landmarks will be used to initialize the bounding rotated rectangle (ROI) in the next frame
                    self.hand_from_landmarks = mpu.hand_landmarks_to_rect(self.hands[0])
                    self.use_previous_landmarks = True
                else:
                    self.use_previous_landmarks = False

            # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
            for hand in self.hands:
                if self.pad_h > 0:
                    hand.landmarks[:,1] -= self.pad_h
                    for i in range(len(hand.rect_points)):
                        hand.rect_points[i][1] -= self.pad_h
                if self.pad_w > 0:
                    hand.landmarks[:,0] -= self.pad_w
                    for i in range(len(hand.rect_points)):
                        hand.rect_points[i][0] -= self.pad_w
        else: # not use_lm
            if self.xyz:
                self.query_xyz(self.spatial_loc_roi_from_palm_center)

        return video_frame, self.hands


    def exit(self):
        self.device.close()
        # Print some stats
        if self.stats:
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {self.fps.nb_frames()})")
            print(f"# palm detection inferences received : {self.nb_pd_inferences}")
            if self.use_lm: print(f"# hand landmark inferences received  : {self.nb_lm_inferences}")
            if self.input_type != "rgb":
                print(f"Palm detection round trip            : {self.glob_pd_rtrip_time/self.nb_pd_inferences*1000:.1f} ms")
                print(f"Hand landmark round trip             : {self.glob_lm_rtrip_time/self.nb_lm_inferences*1000:.1f} ms")
            if self.xyz:
                print(f"Spatial location requests round trip : {self.glob_spatial_rtrip_time/self.nb_anchors*1000:.1f} ms")           
