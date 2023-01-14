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
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")
MOVENET_LIGHTNING_MODEL = str(SCRIPT_DIR / "models/movenet_singlepose_lightning_U8_transpose.blob")
MOVENET_THUNDER_MODEL = str(SCRIPT_DIR / "models/movenet_singlepose_thunder_U8_transpose.blob")

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1)#.flatten()



class HandTrackerBpf:
    """
    Mediapipe Hand Tracker for depthai
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host (Edge mode only),
                    - a file path of an image or a video,
                    - an integer (eg 0) for a webcam id,
    - pd_model: palm detection model blob file,
    - pd_nms_thresh: NMS threshold,
    - pd_score: confidence score to determine whether a detection is reliable (a float between 0 and 1).
    - use_lm: boolean. When True, run landmark model. Otherwise, only palm detection model is run
    - lm_model: landmark model. Either:
                    - 'full' for LANDMARK_MODEL_FULL,
                    - 'lite' for LANDMARK_MODEL_LITE,
                    - 'sparse' for LANDMARK_MODEL_SPARSE,
                    - a path of a blob file. 
    - lm_score_thresh : confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1).
    - use_world_landmarks: boolean. The landmarks model yields 2 types of 3D coordinates : 
                    - coordinates expressed in pixels in the image, always stored in hand.landmarks,
                    - coordinates expressed in meters in the world, stored in hand.world_landmarks 
                    only if use_world_landmarks is True.
    - solo: boolean, when True detect one hand max (much faster since we run the pose detection model only if no hand was detected in the previous frame)
    - xyz : boolean, when True get the (x, y, z) coords of the detected hands (if the device supports depth measure).
    - crop : boolean which indicates if square cropping on source images is applied or not
    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - resolution : sensor resolution "full" (1920x1080) or "ultra" (3840x2160),
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                    The width is calculated accordingly to height and depends on value of 'crop'
    - use_gesture : boolean, when True, recognize hand poses froma predefined set of poses
                    (ONE, TWO, THREE, FOUR, FIVE, OK, PEACE, FIST)
    - body_pre_focusing: "right" or "left" or "group" or "higher". Body pre focusing is the use
                    of a body pose detector to help to focus on the region of the image that
                    contains one hand ("left" or "right") or "both" hands. 
                    If not in solo mode, body_pre_focusing is forced to 'group'
    - body_model : Movenet single pose model: "lightning", "thunder"
    - body_score_thresh : Movenet score thresh
    - hands_up_only: boolean. When using body_pre_focusing, if hands_up_only is True, consider only hands for which the wrist keypoint
                    is above the elbow keypoint.
    - single_hand_tolerance_thresh (Duo mode only) : In Duo mode, if there is only one hand in a frame, 
                    in order to know when a second hand will appear you need to run the palm detection 
                    in the following frames. Because palm detection is slow, you may want to delay 
                    the next time you will run it. 'single_hand_tolerance_thresh' is the number of 
                    frames during only one hand is detected before palm detection is run again.
    - lm_nb_threads : 1 or 2 (default=2), number of inference threads for the landmark model
    - stats : boolean, when True, display some statistics when exiting.   
    - trace : int, 0 = no trace, otherwise print some debug messages or show output of ImageManip nodes
            if trace & 1, print application level info like number of palm detections  
    """
    def __init__(self, input_src=None,
                pd_model=PALM_DETECTION_MODEL, 
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                use_lm=True,
                lm_model="lite",
                lm_score_thresh=0.5,
                use_world_landmarks=False,
                solo=False,
                xyz=False,
                crop=False,
                internal_fps=23,
                resolution="full",
                internal_frame_height=640,
                use_gesture=False,
                body_pre_focusing = 'group',
                body_model = "thunder",
                body_score_thresh=0.2,
                hands_up_only=True,
                single_hand_tolerance_thresh=10,
                lm_nb_threads=2,
                stats=False,
                trace=0
                ):

        self.pd_model = pd_model
        print(f"Palm detection blob : {self.pd_model}")
        if use_lm:
            if lm_model == "full":
                self.lm_model = LANDMARK_MODEL_FULL
            elif lm_model == "lite":
                self.lm_model = LANDMARK_MODEL_LITE
            elif lm_model == "sparse":
                self.lm_model = LANDMARK_MODEL_SPARSE
            else:
                self.lm_model = lm_model
            print(f"Landmark blob       : {self.lm_model}")
        if not use_lm and solo:
            print("Warning: solo mode desactivated when not using landmarks")
            self.solo = False
        else:
            self.solo = solo
        if self.solo:
            print("In Solo mode, # of landmark model threads is forced to 1")
            self.lm_nb_threads = 1
            self.body_pre_focusing = body_pre_focusing 
        else:
            assert lm_nb_threads in [1, 2]
            self.lm_nb_threads = lm_nb_threads
            print("In Duo mode, body_pre_focusing is forced to 'group'")
            self.body_pre_focusing = "group"
        self.body_score_thresh = body_score_thresh
        if body_model == "lightning":
            self.body_model = MOVENET_LIGHTNING_MODEL
            self.body_input_length = 192 
        else:
            self.body_model = MOVENET_THUNDER_MODEL
            self.body_input_length = 256 
        print(f"Body pose blob      : {self.body_model}")
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_score_thresh = lm_score_thresh
        
        self.xyz = False
        self.crop = crop 
        self.use_world_landmarks = use_world_landmarks
        self.internal_fps = internal_fps     
        self.stats = stats
        self.trace = trace
        self.use_gesture = use_gesture
        self.single_hand_tolerance_thresh = single_hand_tolerance_thresh

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
                input_src = int(input_src)
            self.cap = cv2.VideoCapture(input_src)
            self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.img_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.img_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("Video FPS:", self.video_fps)
        
        if self.input_type != "rgb":
            print(f"Original frame size: {self.img_w}x{self.img_h}")
            if self.crop:
                self.frame_size = min(self.img_w, self.img_h)
            else:
                self.frame_size = max(self.img_w, self.img_h)
            self.crop_w = max((self.img_w - self.frame_size) // 2, 0)
            if self.crop_w: print("Cropping on width :", self.crop_w)
            self.crop_h = max((self.img_h - self.frame_size) // 2, 0)
            if self.crop_h: print("Cropping on height :", self.crop_h)

            self.pad_w = max((self.frame_size - self.img_w) // 2, 0)
            if self.pad_w: print("Padding on width :", self.pad_w)
            self.pad_h = max((self.frame_size - self.img_h) // 2, 0)
            if self.pad_h: print("Padding on height :", self.pad_h)
                     
            if self.crop: self.img_h = self.img_w = self.frame_size
            print(f"Frame working size: {self.img_w}x{self.img_h}")
        
        self.bpf = mpu.BodyPreFocusing(
            self.img_w, self.img_h, 
            self.pad_w, self.pad_h, 
            self.frame_size,
            mode = self.body_pre_focusing,
            score_thresh=body_score_thresh, 
            hands_up_only=hands_up_only
            )
        self.crop_region = self.bpf.init_crop_region
        self.previous_handedness = None
        self.nb_bpf_inferences = 0
        self.glob_bpf_rtrip_time = 0

        # Create SSD anchors 
        self.pd_input_length = 128 # Palm detection
        self.anchors = mpu.generate_handtracker_anchors(self.pd_input_length, self.pd_input_length)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Define and start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues 
        if self.input_type == "rgb":
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            self.q_bpf_out = self.device.getOutputQueue(name="bpf_out", maxSize=1, blocking=False)
            self.q_pd_in = self.device.getInputQueue(name="pd_in")
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)
            self.q_manip_cfg = self.device.getInputQueue(name="manip_cfg")
            if self.use_lm:
                self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
                self.q_lm_in = self.device.getInputQueue(name="lm_in")
            if self.xyz:
                self.q_spatial_data = self.device.getOutputQueue(name="spatial_data_out", maxSize=4, blocking=False)
                self.q_spatial_config = self.device.getInputQueue("spatial_calc_config_in")

        else:
            self.q_bpf_in = self.device.getInputQueue(name="bpf_in")
            self.q_bpf_out = self.device.getOutputQueue(name="bpf_out", maxSize=4, blocking=True)
            self.q_pd_in = self.device.getInputQueue(name="pd_in")
            self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
            if self.use_lm:
                self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
                self.q_lm_in = self.device.getInputQueue(name="lm_in")

        self.fps = FPS()

        self.nb_frames_pd_inference = 0
        self.nb_frames_lm_inference = 0
        self.nb_lm_inferences = 0
        self.nb_failed_lm_inferences = 0
        self.nb_frames_lm_inference_after_landmarks_ROI = 0
        self.nb_frames_no_hand = 0
        self.nb_spatial_requests = 0
        self.glob_pd_rtrip_time = 0
        self.glob_lm_rtrip_time = 0
        self.glob_spatial_rtrip_time = 0

        self.use_previous_landmarks = False
        self.hands_from_landmarks = None
        self.nb_hands_in_previous_frame = 0
        if not self.solo: self.single_hand_count = 0

        
    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)

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
            # Movenet takes RGB input
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

            manip = pipeline.createImageManip()
            manip.setMaxOutputFrameSize(self.body_input_length*self.body_input_length*3)
            manip.setWaitForConfigInput(True)
            manip.inputImage.setQueueSize(1)
            manip.inputImage.setBlocking(False)
            cam.preview.link(manip.inputImage)
            if self.crop:
                cam.setVideoSize(self.frame_size, self.frame_size)
                cam.setPreviewSize(self.frame_size, self.frame_size)               
            else: 
                cam.setVideoSize(self.img_w, self.img_h)
                cam.setPreviewSize(self.img_w, self.img_h)

            manip_cfg_in = pipeline.createXLinkIn()
            manip_cfg_in.setStreamName("manip_cfg")
            manip_cfg_in.out.link(manip.inputConfig)

            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            cam_out.input.setQueueSize(1)
            cam_out.input.setBlocking(False)
            cam.video.link(cam_out.input)

            if self.xyz:
                print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes...")
                # For now, RGB needs fixed focus to properly align with depth.
                # The value used during calibration should be used here
                calib_data = self.device.readCalibration()
                calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.RGB)
                print(f"RGB calibration lens position: {calib_lens_pos}")
                cam.initialControl.setManualFocus(calib_lens_pos)

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

        # Create body pose model
        print("Creating Body Pose Neural Network...")
        bpf_nn = pipeline.createNeuralNetwork()
        bpf_nn.setBlobPath(self.body_model)
        if self.input_type == "rgb":
            bpf_nn.input.setQueueSize(1)
            bpf_nn.input.setBlocking(False)
            manip.out.link(bpf_nn.input)
        else:
            bpf_in = pipeline.createXLinkIn()
            bpf_in.setStreamName("bpf_in")
            bpf_in.out.link(bpf_nn.input)

        # Body pose output
        bpf_out = pipeline.createXLinkOut()
        bpf_out.setStreamName("bpf_out")
        bpf_nn.out.link(bpf_out.input)

        # Define palm detection model
        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(self.pd_model)
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)

        # Palm detection input        
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
            lm_nn.setNumInferenceThreads(self.lm_nb_threads)
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
   
    def bpf_postprocess(self, inference):
        kps = np.array(inference.getLayerFp16('Identity')).reshape(-1,3) # 17x3
        body = mpu.Body(
                    scores=kps[:,2], 
                    keypoints_norm=kps[:,[1,0]], 
                    score_thresh=self.body_score_thresh, 
                    crop_region=self.crop_region)
        body.next_crop_region = self.bpf.determine_crop_region(body)
        focus_zone, hand_zone_label = self.bpf.get_focus_zone(body)
        return focus_zone, hand_zone_label, body

    def pd_postprocess(self, inference, focus_zone):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,18)) # 896x18
        # Decode bboxes
        hands = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=self.solo)
        # Non maximum suppression (not needed if solo)
        if not self.solo:
            hands = mpu.non_max_suppression(hands, self.pd_nms_thresh)
        if focus_zone:
            zone_size = focus_zone[2] - focus_zone[0]
            for h in hands:
                box = h.pd_box * zone_size
                box[0] += focus_zone[0] 
                box[1] += focus_zone[1]
                h.pd_box = box / self.frame_size
                for kp in h.pd_kps:
                    kp *= zone_size
                    kp += focus_zone[0:2]
                    kp /= self.frame_size
        if self.use_lm:
            mpu.detections_to_rect(hands)
            mpu.rect_transformation(hands, self.frame_size, self.frame_size)
        return hands

    def lm_postprocess(self, hand, inference):
        # print(inference.getAllLayerNames())
        # The output names of the landmarks model are :
        # Identity_1 (1x1) : score 
        # Identity_2 (1x1) : handedness
        # Identity_3 (1x63) : world 3D landmarks (in meters)
        # Identity (1x63) : screen 3D landmarks (in pixels)
        hand.lm_score = inference.getLayerFp16("Identity_1")[0]  
        if hand.lm_score > self.lm_score_thresh:  
            hand.handedness = inference.getLayerFp16("Identity_2")[0]
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
            hand.landmarks = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int32)

            # World landmarks
            if self.use_world_landmarks:
                hand.world_landmarks = np.array(inference.getLayerFp16("Identity_3_dense/BiasAdd/Add")).reshape(-1,3)

            if self.use_gesture: mpu.recognize_gesture(hand)
            
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
    def smart_crop_and_resize(self, frame, crop_region):
        """Crops and resize the image to prepare for the model input."""
        cropped = frame[max(0,crop_region.ymin):min(self.img_h,crop_region.ymax), max(0,crop_region.xmin):min(self.img_w,crop_region.xmax)]
        
        if crop_region.xmin < 0 or crop_region.xmax >= self.img_w or crop_region.ymin < 0 or crop_region.ymax >= self.img_h:
            # Padding is necessary        
            cropped = cv2.copyMakeBorder(cropped, 
                                        max(0,-crop_region.ymin), 
                                        max(0, crop_region.ymax-self.img_h),
                                        max(0,-crop_region.xmin), 
                                        max(0, crop_region.xmax-self.img_w),
                                        cv2.BORDER_CONSTANT)

        cropped = cv2.resize(cropped, (self.body_input_length, self.body_input_length), interpolation=cv2.INTER_AREA)
        return cropped

    def next_frame(self):

        
        body = None
        hand_zone_label = None
        bag = {}
        bag["body"] = None
        self.fps.update()
        focus_zone = None # Used with body pre focusing
        if self.input_type == "rgb":
            if not self.use_previous_landmarks:
                # Send image manip config to the device                
                # We prepare the input to Movenet = Movenet smart cropping  
                cfg = dai.ImageManipConfig()                 
                points = [
                [self.crop_region.xmin, self.crop_region.ymin],
                [self.crop_region.xmax-1, self.crop_region.ymin],
                [self.crop_region.xmax-1, self.crop_region.ymax-1],
                [self.crop_region.xmin, self.crop_region.ymax-1]]
                point2fList = []
                for p in points:
                    pt = dai.Point2f()
                    pt.x, pt.y = p[0], p[1]
                    point2fList.append(pt)
                cfg.setWarpTransformFourPoints(point2fList, False)
                cfg.setResize(self.body_input_length, self.body_input_length)
                cfg.setFrameType(dai.ImgFrame.Type.RGB888p)

                self.q_manip_cfg.send(cfg)

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
                    return None, None, None
            # Cropping and/or padding of the video frame
            video_frame = frame[self.crop_h:self.crop_h+self.frame_size, self.crop_w:self.crop_w+self.frame_size]
            if self.pad_h or self.pad_w:
                square_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
            else:
                square_frame = video_frame
        
            if not self.use_previous_landmarks:
                smart_cropped = self.smart_crop_and_resize(video_frame, self.crop_region)
                smart_cropped = cv2.cvtColor(smart_cropped, cv2.COLOR_BGR2RGB)
                frame_nn = dai.ImgFrame()
                frame_nn.setTimestamp(time.monotonic())
                frame_nn.setWidth(self.body_input_length)
                frame_nn.setHeight(self.body_input_length)
                frame_nn.setData(to_planar(smart_cropped, (self.body_input_length, self.body_input_length)))
                self.q_bpf_in.send(frame_nn)
                bpf_rtrip_time = now()

        if not self.use_previous_landmarks:
            # Get body pose
            inference = self.q_bpf_out.get()
            focus_zone, hand_zone_label, body = self.bpf_postprocess(inference)
            if self.trace & 1:
                print(f"Body pose - focus zone: {None if focus_zone is None else hand_zone_label}")
            self.crop_region = body.next_crop_region
            self.nb_bpf_inferences += 1
            bag["bpf_inference"] = 1
            bag["body"] = body          
            if focus_zone:
                bag["focus_zone"] = focus_zone.copy() # Saved for rendering
                # focus_zone is calculated in the original image
                # We convert it in the padded square image
                focus_zone[0] += self.pad_w
                focus_zone[1] += self.pad_h
                focus_zone[2] += self.pad_w
                focus_zone[3] += self.pad_h
                focus_frame = square_frame[focus_zone[1]:focus_zone[3], focus_zone[0]:focus_zone[2]]
                if self.trace & 2:
                    cv2.imshow("BPF frame", focus_frame)
                if self.input_type != "rgb": 
                    self.glob_bpf_rtrip_time += now() - bpf_rtrip_time
                # Send image to pd_nn
                frame_nn = dai.ImgFrame()
                frame_nn.setTimestamp(time.monotonic())
                frame_nn.setWidth(self.pd_input_length)
                frame_nn.setHeight(self.pd_input_length)
                frame_nn.setData(to_planar(focus_frame if focus_zone else square_frame, (self.pd_input_length, self.pd_input_length)))
                self.q_pd_in.send(frame_nn)
                pd_rtrip_time = now()
            else:
                self.nb_frames_no_hand += 1
                return video_frame, [], bag
           

        # Get palm detection
        if self.use_previous_landmarks:
            self.hands = self.hands_from_landmarks
        else:
            inference = self.q_pd_out.get()
            if self.input_type != "rgb": 
                self.glob_pd_rtrip_time += now() - pd_rtrip_time
            hands = self.pd_postprocess(inference, focus_zone)
            if self.trace & 1:
                print(f"Palm detection - nb hands detected: {len(hands)}")
            self.nb_frames_pd_inference += 1  
            bag["pd_inference"] = 1 
            if not self.solo and self.nb_hands_in_previous_frame == 1 and len(hands) <= 1:
                self.hands = self.hands_from_landmarks
            else:
                self.hands = hands

        if len(self.hands) == 0: self.nb_frames_no_hand += 1

        # Hand landmarks, send requests
        if self.use_lm:
            nb_lm_inferences = len(self.hands)
            if self.use_previous_landmarks: self.nb_frames_lm_inference_after_landmarks_ROI += 1
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
            bag["lm_inference"] = len(self.hands)
            self.hands = [ h for h in self.hands if h.lm_score > self.lm_score_thresh]

            if self.trace & 1:
                print(f"Landmarks - nb hands detected : {len(self.hands)}")

            # Check that 2 detected hands do not correspond to the same hand in the image
            # That may happen when one hand in the image cross another one
            # A simple method is to assure that the center of the rotated rectangles are not too close
            if len(self.hands) == 2: 
                dist_rect_centers = mpu.distance(np.array((self.hands[0].rect_x_center_a, self.hands[0].rect_y_center_a)), np.array((self.hands[1].rect_x_center_a, self.hands[1].rect_y_center_a)))
                if dist_rect_centers < 5:
                    # Keep the hand with higher landmark score
                    if self.hands[0].lm_score > self.hands[1].lm_score:
                        self.hands = [self.hands[0]]
                    else:
                        self.hands = [self.hands[1]]
                    if self.trace & 1: print("!!! Removing one hand because too close to the other one")

            nb_hands = len(self.hands)

            if self.xyz:
                self.query_xyz(self.spatial_loc_roi_from_wrist_landmark)

            if self.solo:
                if nb_hands == 1:
                    # hand_from_landmarks will be used to initialize the bounding rotated rectangle (ROI) in the next frame
                    self.hands_from_landmarks = [mpu.hand_landmarks_to_rect(self.hands[0])]
                    self.use_previous_landmarks = True
                    # In Body Pre Focusing mode, set handedness from hand_zone_label calculated from bpf_postprocess()  
                    # (much more reliable than inferred handedness from landmark model)
                    if hand_zone_label is None and self.previous_handedness is not None:
                        self.hands[0].handedness = self.previous_handedness
                    elif hand_zone_label in ["right", "left"]:
                        self.previous_handedness = self.hands[0].handedness = 1 if hand_zone_label == "right" else 0                         
                else:
                    self.use_previous_landmarks = False
                    self.previous_handedness = None
            else: # Duo
                if not self.use_previous_landmarks: 
                    if hand_zone_label is not None:
                        if nb_hands == 2:
                            if hand_zone_label == "group": # "group" -> we expect 2 hands
                                d_h0_right =  body.distance_to_wrist(self.hands[0], "right", pad_h=self.pad_h)
                                d_h1_right =  body.distance_to_wrist(self.hands[1], "right", pad_h=self.pad_h)
                                d_h0_left =  body.distance_to_wrist(self.hands[0], "left", pad_h=self.pad_h)
                                d_h1_left =  body.distance_to_wrist(self.hands[1], "left", pad_h=self.pad_h)
                                if d_h0_left + d_h1_right < d_h0_right + d_h1_left:
                                    self.hands[0].handedness = 0
                                    self.hands[1].handedness = 1
                                else:
                                    self.hands[0].handedness = 1
                                    self.hands[1].handedness = 0
                            else: # hand_zone_label == "left" or "right". Normally we should have only 1 hand but just ine case...
                                d_h0 = body.distance_to_wrist(self.hands[0], hand_zone_label, pad_h=self.pad_h)
                                d_h1 = body.distance_to_wrist(self.hands[1], hand_zone_label, pad_h=self.pad_h)
                                if d_h0 < d_h1:
                                    self.hands[0].handedness = 1 if hand_zone_label == "right" else 0
                                    # We arbitrary decide that the other hand has the other handedness
                                    self.hands[1].handedness = 1 - self.hands[0].handedness
                                else:
                                    self.hands[1].handedness = 1 if hand_zone_label == "right" else 0
                                    # We arbitrary decide that the other hand has the other handedness
                                    self.hands[0].handedness = 1 - self.hands[0].handedness
                            self.previous_handedness = [self.hands[0].handedness, self.hands[1].handedness]
                            self.hands_from_landmarks = [mpu.hand_landmarks_to_rect(self.hands[0]), mpu.hand_landmarks_to_rect(self.hands[1])]
                            self.use_previous_landmarks = True
                        elif nb_hands == 1:
                            if hand_zone_label == "group": # We would have expected 2 hands 
                                d_h0_right =  body.distance_to_wrist(self.hands[0], "right", pad_h=self.pad_h)
                                d_h0_left =  body.distance_to_wrist(self.hands[0], "left", pad_h=self.pad_h)
                                self.hands[0].handedness = 1 if d_h0_right < d_h0_left else 0
                            else: # hand_zone_label == "left" or "right".
                                self.hands[0].handedness = 1 if hand_zone_label == "right" else 0
                            self.previous_handedness = [self.hands[0].handedness]
                            self.hands_from_landmarks = [mpu.hand_landmarks_to_rect(self.hands[0])]
                            self.use_previous_landmarks = True
                        else: # nb_hands == 0
                            self.use_previous_landmarks = False
                elif nb_hands != self.nb_hands_in_previous_frame:   
                    # We ask for body detection for the next frame
                    self.use_previous_landmarks = False   
                    # ...but there is a chance that we don't use body detection result 
                    # if there is only one hand
                    if nb_hands == 1: 
                        # Actually we have also nb_hands_in_previous_frame = 2
                        # The current detected hand was detected from one of the 2 hands_from_landmarks
                        # Which one ? The one which has its handedness (or landmarks or norm_landmarks)
                        # attribute filled
                        hand_i =  0 if hasattr(self.hands_from_landmarks[0], "handedness") else 1
                        self.hands[0].handedness = self.previous_handedness[hand_i]
                        self.hands_from_landmarks = [mpu.hand_landmarks_to_rect(self.hands[0])] 
                else:
                    if nb_hands == 2:
                        self.hands[0].handedness = self.previous_handedness[0]
                        self.hands[1].handedness = self.previous_handedness[1]
                        self.hands_from_landmarks = [mpu.hand_landmarks_to_rect(self.hands[0]), mpu.hand_landmarks_to_rect(self.hands[1])]
                        # self.use_previous_landmarks = True
                    elif nb_hands == 1:
                        self.hands[0].handedness = self.previous_handedness[0]
                        self.hands_from_landmarks = [mpu.hand_landmarks_to_rect(self.hands[0])] 
                        if self.single_hand_count >= self.single_hand_tolerance_thresh:
                            self.use_previous_landmarks = False
                            self.single_hand_count = 0
                        else:
                            self.single_hand_count += 1
                            # self.use_previous_landmarks = True
                    else:
                        self.use_previous_landmarks = False
                        
            # Stats
            if nb_lm_inferences: self.nb_frames_lm_inference += 1
            self.nb_lm_inferences += nb_lm_inferences
            self.nb_failed_lm_inferences += nb_lm_inferences - nb_hands 
            
            for hand in self.hands:
                # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
                if self.pad_h > 0:
                    hand.landmarks[:,1] -= self.pad_h
                    for i in range(len(hand.rect_points)):
                        hand.rect_points[i][1] -= self.pad_h
                if self.pad_w > 0:
                    hand.landmarks[:,0] -= self.pad_w
                    for i in range(len(hand.rect_points)):
                        hand.rect_points[i][0] -= self.pad_w

                # Set the hand label
                hand.label = "right" if hand.handedness > 0.5 else "left"  

            self.nb_hands_in_previous_frame = nb_hands     

        else: # not use_lm
            if self.xyz:
                self.query_xyz(self.spatial_loc_roi_from_palm_center)


        return video_frame, self.hands, bag

    def exit(self):
        self.device.close()
        # Print some stats
        if self.stats:
            nb_frames = self.fps.nb_frames()
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {nb_frames})")
            print(f"# frames w/ no hand           : {self.nb_frames_no_hand} ({100*self.nb_frames_no_hand/nb_frames:.1f}%)")
            print(f"# frames w/ palm detection    : {self.nb_frames_pd_inference} ({100*self.nb_frames_pd_inference/nb_frames:.1f}%)")
            print(f"# frames w/ landmark inference : {self.nb_frames_lm_inference} ({100*self.nb_frames_lm_inference/nb_frames:.1f}%)- # after palm detection: {self.nb_frames_lm_inference - self.nb_frames_lm_inference_after_landmarks_ROI} - # after landmarks ROI prediction: {self.nb_frames_lm_inference_after_landmarks_ROI}")
            if not self.solo:
                print(f"On frames with at least one landmark inference, average number of landmarks inferences/frame: {self.nb_lm_inferences/self.nb_frames_lm_inference:.2f}")
            print(f"# lm inferences: {self.nb_lm_inferences} - # failed lm inferences: {self.nb_failed_lm_inferences} ({100*self.nb_failed_lm_inferences/self.nb_lm_inferences:.1f}%)")
            
            if self.input_type != "rgb":
                print(f"Body pose estimation round trip      : {self.glob_bpf_rtrip_time/self.nb_bpf_inferences*1000:.1f} ms")
                print(f"Palm detection round trip            : {self.glob_pd_rtrip_time/self.nb_frames_pd_inference*1000:.1f} ms")
                if self.use_lm and self.nb_lm_inferences:
                    print(f"Hand landmark round trip             : {self.glob_lm_rtrip_time/self.nb_frames_lm_inference*1000:.1f} ms")
            if self.xyz:
                print(f"Spatial location requests round trip : {self.glob_spatial_rtrip_time/self.nb_spatial_requests*1000:.1f} ms")           
