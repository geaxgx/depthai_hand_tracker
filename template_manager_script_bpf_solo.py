"""
This file is the template of the scripting node source code in edge mode
Substitution is made in HandTrackerEdge.py

In the following:
rrn_ : normalized [0:1] coordinates in rotated rectangle coordinate systems 
sqn_ : normalized [0:1] coordinates in squared input image
"""
import marshal
from math import sin, cos, atan2, pi, degrees, floor, dist


pad_h = ${_pad_h}
img_h = ${_img_h}
img_w = ${_img_w}
frame_size = ${_frame_size}
crop_w = ${_crop_w}

${_TRACE1} ("Starting manager script node")

# Note that the output of the movenet model is a list named 'body' of 17*3 elements
# The information for the ith keypoint is :
# y coord: body[3*i]
# x coord: body[1+3*i] 
# score  : body[2+3*i]
body_score_thresh=${_body_score_thresh}
hands_up_only = ${_hands_up_only}

BODY_KP = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

torso_joints = [BODY_KP["left_hip"], BODY_KP["right_hip"], BODY_KP["left_shoulder"], BODY_KP["right_shoulder"]]

# For Movenet smart croppping, defines the default crop region (pads the full image from both sides to make it a square image) 
# Used when the algorithm cannot reliably determine the crop region from the previous frame.
init_crop_region = {'xmin': 0, 'ymin': -pad_h, 'xmax': frame_size, 'ymax': -pad_h+frame_size, 'size': frame_size}
crop_region = init_crop_region

def torso_visible(scores):
    # Checks whether there are enough torso keypoints.
    # This function checks whether the model is confident at predicting one of the
    # shoulders/hips which is required to determine a good crop region.

    return ((scores[BODY_KP["left_hip"]] > body_score_thresh or
            scores[BODY_KP["right_hip"]] > body_score_thresh) and
            (scores[BODY_KP["left_shoulder"]] > body_score_thresh or
            scores[BODY_KP["right_shoulder"]] > body_score_thresh))

def determine_torso_and_body_range(x, y, scores, center_x, center_y):
    # Calculates the maximum distance from each keypoints to the center location.
    # The function returns the maximum distances from the two sets of keypoints:
    # full 17 keypoints and 4 torso keypoints. The returned information will be
    # used to determine the crop size. See determine_crop_region for more detail.

    max_torso_yrange = 0.0
    max_torso_xrange = 0.0
    for i in torso_joints:
        dist_y = abs(center_y - y[i])
        dist_x = abs(center_x - x[i])
        if dist_y > max_torso_yrange:
            max_torso_yrange = dist_y
        if dist_x > max_torso_xrange:
            max_torso_xrange = dist_x

    max_body_yrange = 0.0
    max_body_xrange = 0.0
    for i in range(17):
        if scores[i] < body_score_thresh:
            continue
        dist_y = abs(center_y - y[i])
        dist_x = abs(center_x - x[i])
        if dist_y > max_body_yrange:
            max_body_yrange = dist_y
        if dist_x > max_body_xrange:
            max_body_xrange = dist_x

    return [max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange]

def determine_crop_region(scores, x, y):
    # Determines the region to crop the image for the model to run inference on.
    # The algorithm uses the detected joints from the previous frame to estimate
    # the square region that encloses the full body of the target person and
    # centers at the midpoint of two hip joints. The crop size is determined by
    # the distances between each joints and the center point.
    # When the model is not confident with the four torso joint predictions, the
    # function returns a default crop which is the full image padded to square.

    if torso_visible(scores):
        center_x = (x[11] + x[12]) // 2
        center_y = (y[11] + y[12]) // 2
        max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange = determine_torso_and_body_range(x, y, scores, center_x, center_y)
        crop_length_half = max(max_torso_xrange * 1.9, max_torso_yrange * 1.9, max_body_yrange * 1.2, max_body_xrange * 1.2)
        crop_length_half = int(round(min(crop_length_half, max(center_x, img_w - center_x, center_y, img_h - center_y))))
        crop_corner = [center_x - crop_length_half, center_y - crop_length_half]

        if crop_length_half > max(img_w, img_h) / 2:
            return init_crop_region
        else:
            crop_length = crop_length_half * 2
            return {'xmin': crop_corner[0], 'ymin': crop_corner[1], 'xmax': crop_corner[0]+crop_length, 'ymax': crop_corner[1]+crop_length, 'size': crop_length}
    else:
        return init_crop_region

def movenet_postprocess(body, crop_region):
    size = crop_region['size']
    xmin = crop_region['xmin']
    ymin = crop_region['ymin']
    scores = []
    x = []
    y = []
    for i in range(17):
        xn = body[3*i+1]
        yn = body[3*i]
        scores.append(body[3*i+2])
        # x any are the body keypoint coordinates in the source image in pixels
        x.append(int(xmin + xn * size)) 
        y.append(int(ymin + yn * size)) 
          
    next_crop_region = determine_crop_region(scores, x, y) 
    return x, y, scores, next_crop_region

def estimate_focus_zone_size(scale=1.0):
    segments = [
        ("left_shoulder", "left_elbow", 2.3),
        ("left_elbow", "left_wrist", 2.3),
        ("left_shoulder", "left_hip", 1),
        ("left_shoulder", "right_shoulder", 1.5),
        ("right_shoulder", "right_elbow", 2.3),
        ("right_elbow", "right_wrist", 2.3),
        ("right_shoulder", "right_hip", 1),
    ]
    lengths = []
    for s in segments:
        id0 = BODY_KP[s[0]]
        id1 = BODY_KP[s[1]]
        if body_scores[id0] > body_score_thresh and body_scores[id1] > body_score_thresh:
            
            l = dist((body_x[id0],body_y[id0]), (body_x[id1],body_y[id1]))
            lengths.append(l)
    if lengths:   
        if ( body_scores[BODY_KP["left_hip"]] < body_score_thresh and
            body_scores[BODY_KP["right_hip"]] < body_score_thresh or
            body_scores[BODY_KP["left_shoulder"]] < body_score_thresh and
            body_scores[BODY_KP["right_shoulder"]] < body_score_thresh) :
            coef = 1.0
        else:
            coef = 1.0
        return 2 * int(coef * scale * max(lengths) / 2) # The size is made even
        # return coef * scale * max(lengths)  
    else:
        return 0

def zone_from_center_size(x, y, size):
    # Return zone [left, top, right, bottom] 
    # from zone center (x,y) and zone size (the zone is square).
    half_size = size // 2
    size = half_size * 2
    if size > img_w:
        x = img_w // 2
    x1 = x - half_size
    if x1 < 0:
        x1 = 0
    elif x1 + size > img_w:
        x1 = img_w - size
    x2 = x1 + size
    if size > img_h: 
        y = img_h // 2
    y1 = y - half_size
    if y1 < -pad_h: 
        y1 = -pad_h
    elif y1 + size > img_h + pad_h:
        y1 = img_h + pad_h - size
    y2 = y1 + size
    return [x1, y1, x2, y2]

def get_one_hand_zone(hand_label):
    # Return the zone [left, top, right, bottom] around the hand given by its label "hand_label" ("left" or "right")
    # Values are expressed in pixels in the source image C.S.
    # If the wrist keypoint is not visible, return None.
    # If self.hands_up_only is True, return None if wrist keypoint is below elbow keypoint.
    wrist_kp = hand_label + "_wrist"
    id_wrist = BODY_KP[wrist_kp]
    if body_scores[id_wrist] < body_score_thresh: 
        return None
    x = body_x[id_wrist]
    y = body_y[id_wrist]
    if hands_up_only:
        # We want to detect only hands where the wrist is above the elbow (when visible)
        elbow_kp = hand_label + "_elbow"
        id_elbow = BODY_KP[elbow_kp]
        if body_scores[id_elbow] > body_score_thresh and \
            body_y[id_elbow] < y:
            return None
    # Let's evaluate the size of the focus zone
    size = estimate_focus_zone_size()
    if size == 0: return [0, -pad_h, frame_size, frame_size-pad_h]
    return zone_from_center_size(x, y, size)


def get_focus_zone(hand_label):
    #  Return a list [focus_zone, label]
    # 'focus_zone' is a zone around a hand or hands, depending on the value 
    # of hand_label ("left", "right", "higher" or "group") and on the value of self.hands_up_only.
    #     - hand_label = "left" (resp "right"): we are looking for the zone around the left (resp right) wrist,
    #     - hand_label = "group": the zone encompasses both wrists,
    #     - hand_label = "higher": the zone is around the higher wrist (smaller y value),
    #     - hands_up_only = True: we don't take into consideration the wrist if the corresponding elbow is above the wrist,
    # focus_zone is a list [left, top, right, bottom] defining the top-left and right-bottom corners of a square. 
    # Values are expressed in pixels in the source image C.S.
    # The zone is constrained to the squared source image (= source image with padding if necessary). 
    # It means that values can be negative.
    # left and right in [-pad_w, img_w + pad_w]
    # top and bottom in [-pad_h, img_h + pad_h]
    # 'label' describes which wrist keypoint(s) were used to build the zone : "left", "right" or "group" (if built from both wrists)
    
    # If the wrist keypoint(s) is(are) not present or is(are) present but self.hands_up_only = True and
    # wrist(s) is(are) below corresponding elbow(s), then focus_zone = None.
    if hand_label == "group":
        zonel = get_one_hand_zone("left")
        if zonel:
            zoner = get_one_hand_zone("right")
            if zoner:
                xl1, yl1, xl2, yl2 = zonel
                xr1, yr1, xr2, yr2 = zoner
                x1 = min(xl1, xr1)
                y1 = min(yl1, yr1)
                x2 = max(xl2, xr2)
                y2 = max(yl2, yr2)
                # Center (x,y)
                x = (x1+x2)//2
                y = (y1+y2)//2
                size_x = x2-x1
                size_y = y2-y1
                size = max(size_x, size_y) 
                zone_hand = [zone_from_center_size(x, y, size), "group"]
            else:
                zone_hand = [zonel, "left"]
        else:
            zoner = get_one_hand_zone("right")
            if zoner:
                zone_hand = [zoner, "right"]
            else:
                zone_hand = [None, None]

    elif hand_label == "higher":
        id_left_wrist = BODY_KP["left_wrist"]
        id_right_wrist = BODY_KP["right_wrist"]
        if body_scores[id_left_wrist] > body_score_thresh:
            if body_scores[id_right_wrist] > body_score_thresh:
                if body_y[id_left_wrist] > body_y[id_right_wrist]:
                    hand_label = "right"
                else:
                    hand_label = "left"
            else: 
                 hand_label = "left"
        else:
            if body_scores[id_right_wrist] > body_score_thresh:
                hand_label = "right"
            else:
                return [None, None]
        zone = get_one_hand_zone(hand_label)
        if zone: 
            zone_hand = [zone, hand_label]
        else:
            zone_hand = [None, None]
    else: # "left" or "right"
        zone_hand = [get_one_hand_zone(hand_label), hand_label]
    return zone_hand

previous_hand_label = None

# BufferMgr is used to statically allocate buffers once 
# (replace dynamic allocation). 
# These buffers are used for sending result to host
class BufferMgr:
    def __init__(self):
        self._bufs = {}
    def __call__(self, size):
        try:
            buf = self._bufs[size]
        except KeyError:
            buf = self._bufs[size] = Buffer(size)
            ${_TRACE2} (f"New buffer allocated: {size}")
        return buf
        
buffer_mgr = BufferMgr()

def send_result(result):
    result_serial = marshal.dumps(result)
    buffer = buffer_mgr(len(result_serial))  
    buffer.getData()[:] = result_serial  
    node.io['host'].send(buffer)
    ${_TRACE2} ("Manager sent result to host")

# bd_pd_inf: 0, 1 or 2. 
#       0: neither body and palm detections has been run on the frame;
#       1: only body detection run (but no boundy found);
#       2: both body and palm detections run (body found).
# nb_lm_inf: 0 or 1 (or 2 in duo mode). Number of landmark regression inferences on the frame.
# bd_pd_inf=1 or 2 and nb_lm_inf=0 means the body or palm detection hasn't found any hand
# bd_pd_inf, nb_lm_inf are used for statistics 
def send_result_no_hand(bd_pd_inf, nb_lm_inf):
    result = dict([("bd_pd_inf", bd_pd_inf), ("nb_lm_inf", nb_lm_inf)])
    send_result(result)

def send_result_hand(bd_pd_inf, nb_lm_inf, lm_score, handedness, rect_center_x, rect_center_y, rect_size, rotation, rrn_lms, sqn_lms, world_lms, xyz, xyz_zone):
      
    result = dict([("bd_pd_inf", bd_pd_inf), ("nb_lm_inf", nb_lm_inf), ("lm_score", [lm_score]), ("handedness", [handedness]), ("rotation", [rotation]),
            ("rect_center_x", [rect_center_x]), ("rect_center_y", [rect_center_y]), ("rect_size", [rect_size]), 
            ("rrn_lms", [rrn_lms]), ('sqn_lms', [sqn_lms]), ('world_lms', [world_lms]), ("xyz", xyz), ("xyz_zone", xyz_zone)])
    send_result(result)

def rr2img(rrn_x, rrn_y):
    # Convert a point (rrn_x, rrn_y) expressed in normalized rotated rectangle (rrn)
    # into (X, Y) expressed in normalized image (sqn)
    X = sqn_rr_center_x + sqn_rr_size * ((rrn_x - 0.5) * cos_rot + (0.5 - rrn_y) * sin_rot)
    Y = sqn_rr_center_y + sqn_rr_size * ((rrn_y - 0.5) * cos_rot + (rrn_x - 0.5) * sin_rot)
    return X, Y

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))

# send_new_frame_to_branch defines on which branch new incoming frames are sent
# 0 = body detection branch
# 1 = palm detection branch 
# 2 = hand landmark branch
send_new_frame_to_branch = 0

cfg_pre_pd = ImageManipConfig()
cfg_pre_pd.setResizeThumbnail(128, 128, 0, 0, 0)

id_wrist = 0
id_index_mcp = 5
id_middle_mcp = 9
id_ring_mcp =13
ids_for_bounding_box = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]

lm_input_size = 224

while True:
    hand_label = None
    nb_lm_inf = 0
    if send_new_frame_to_branch == 0: # Routing frame to body detection
        cfg_pre_body = ImageManipConfig()
        points = [
            [crop_region['xmin'], crop_region['ymin']],
            [crop_region['xmax']-1, crop_region['ymin']],
            [crop_region['xmax']-1, crop_region['ymax']-1],
            [crop_region['xmin'], crop_region['ymax']-1]]
        point2fList = []
        for p in points:
            pt = Point2f()
            pt.x, pt.y = p[0], p[1]
            point2fList.append(pt)
        cfg_pre_body.setWarpTransformFourPoints(point2fList, False)
        cfg_pre_body.setResize(${_body_input_length}, ${_body_input_length})
        cfg_pre_body.setFrameType(ImgFrame.Type.RGB888p)
        node.io['pre_body_manip_cfg'].send(cfg_pre_body)
        ${_TRACE2} ("Manager sent thumbnail config to pre_body manip")
        # Wait for body detection result 
        body = node.io['from_body_nn'].get().getLayerFp16("Identity")
        ${_TRACE2} ("Manager received result from body_nn")
        # Extract body keypoints and calculate smart crop for next frame
        body_x, body_y, body_scores, crop_region = movenet_postprocess(body, crop_region)
        iwr = BODY_KP['right_wrist']

        # Calculate pre focus zone
        zone, hand_label = get_focus_zone("${_body_pre_focusing}")
        if zone:
            xmin, ymin, xmax, ymax = zone 
            ${_TRACE1} (f"Body pre focusing zone: ({xmin}, {ymin}), ({xmax}, {ymax})")
            points = [
                [xmin,ymin],
                [xmax,ymin],
                [xmax,ymax],
                [xmin,ymax]]
            point2fList = []
            for p in points:
                pt = Point2f()
                pt.x, pt.y = p[0], p[1]
                point2fList.append(pt)
            cfg_pre_pd = ImageManipConfig()
            cfg_pre_pd.setWarpTransformFourPoints(point2fList, False)
            cfg_pre_pd.setResize(128, 128)
            send_new_frame_to_branch = 1
        else:
            ${_TRACE1} (f"Body pre focusing zone: None")
            send_result_no_hand(1, 0)
            continue
 
    if send_new_frame_to_branch == 1: # Routing frame to pd branch
        node.io['pre_pd_manip_cfg'].send(cfg_pre_pd)
        ${_TRACE2} ("Manager sent thumbnail config to pre_pd manip")
        # Wait for pd post processing's result 
        detection = node.io['from_post_pd_nn'].get().getLayerFp16("result")
        ${_TRACE2} ("Manager received pd result (len={len(detection)}) : "+str(detection))
        # detection is list of 2x8 float
        # Currently we keep only the 8 first values as we are in solo mode
        pd_score, box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y = detection[:8]
        
        if pd_score < ${_pd_score_thresh} or box_size < 0:
            send_result_no_hand(2, 0)
            send_new_frame_to_branch = 0
            ${_TRACE1} (f"Palm detection - no hand detected")
            continue
        ${_TRACE1} (f"Palm detection - hand detected")

        if zone:
            # xmin, ymin, xmax are expressed in pixel in the source image C.S.
            # box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y are normalized coords in square zone
            # We need box_x, box_y, box_size, kp0_x, kp0_y, kp2_x, kp2_y expressed in normalized coords in squared source image (sqn_)!
            sqn_zone_size = (xmax - xmin) / frame_size
            box_size *= sqn_zone_size
            sqn_xmin = xmin / frame_size
            sqn_ymin = (ymin + pad_h) /  frame_size
            box_x = sqn_xmin + box_x * sqn_zone_size 
            box_y = sqn_ymin + box_y * sqn_zone_size 
            kp0_x = sqn_xmin + kp0_x * sqn_zone_size
            kp0_y = sqn_ymin + kp0_y * sqn_zone_size
            kp2_x = sqn_xmin + kp2_x * sqn_zone_size
            kp2_y = sqn_ymin + kp2_y * sqn_zone_size

        # scale_center_x = sqn_scale_x - sqn_rr_center_x
        # scale_center_y = sqn_scale_y - sqn_rr_center_y
        kp02_x = kp2_x - kp0_x
        kp02_y = kp2_y - kp0_y
        sqn_rr_size = 2.9 * box_size
        rotation = 0.5 * pi - atan2(-kp02_y, kp02_x)
        rotation = normalize_radians(rotation)
        sqn_rr_center_x = box_x + 0.5*box_size*sin(rotation)
        sqn_rr_center_y = box_y - 0.5*box_size*cos(rotation)

    # Tell pre_lm_manip how to crop hand region 
    rr = RotatedRect()
    rr.center.x    = sqn_rr_center_x
    rr.center.y    = (sqn_rr_center_y * frame_size - pad_h) / img_h
    rr.size.width  = sqn_rr_size
    rr.size.height = sqn_rr_size * frame_size / img_h
    rr.angle       = degrees(rotation)
    cfg = ImageManipConfig()
    cfg.setCropRotatedRect(rr, True)
    cfg.setResize(lm_input_size, lm_input_size)
    node.io['pre_lm_manip_cfg'].send(cfg)
    nb_lm_inf += 1
    ${_TRACE2} ("Manager sent config to pre_lm manip")

    # Wait for lm's result
    lm_result = node.io['from_lm_nn'].get()
    ${_TRACE2} ("Manager received result from lm nn")
    lm_score = lm_result.getLayerFp16("Identity_1")[0]
    if lm_score > ${_lm_score_thresh}:
        handedness = lm_result.getLayerFp16("Identity_2")[0]

        # In Body Pre Focusing mode, set handedness from hand_label calculated from body keypoints 
        # (much more reliable than inferred handedness from landmark model)
        if hand_label is None and previous_hand_label is not None:
            handedness = 1. if previous_hand_label == "right" else 0.
        elif hand_label == "right":
            handedness = 1.
            previous_hand_label = "right"
        elif hand_label == "left":
            handedness = 0.
            previous_hand_label = "left"

        rrn_lms = lm_result.getLayerFp16("Identity_dense/BiasAdd/Add")
        world_lms = 0
        ${_IF_USE_WORLD_LANDMARKS}
        world_lms = lm_result.getLayerFp16("Identity_3_dense/BiasAdd/Add")
        ${_IF_USE_WORLD_LANDMARKS}
        # Retroproject landmarks into the original squared image 
        sqn_lms = []
        cos_rot = cos(rotation)
        sin_rot = sin(rotation)
        for i in range(21):
            rrn_lms[3*i] /= lm_input_size
            rrn_lms[3*i+1] /= lm_input_size
            rrn_lms[3*i+2] /= lm_input_size  #* 0.4
            sqn_x, sqn_y = rr2img(rrn_lms[3*i], rrn_lms[3*i+1])
            sqn_lms += [sqn_x, sqn_y]
        xyz = 0
        xyz_zone = 0
        # Query xyz
        ${_IF_XYZ}
        cfg = SpatialLocationCalculatorConfig()
        conf_data = SpatialLocationCalculatorConfigData()
        conf_data.depthThresholds.lowerThreshold = 100
        conf_data.depthThresholds.upperThreshold = 10000
        zone_size = max(int(sqn_rr_size * frame_size / 10), 8)
        c_x = int(sqn_lms[0] * frame_size -zone_size/2 + crop_w)
        c_y = int(sqn_lms[1] * frame_size -zone_size/2 - pad_h)
        rect_center = Point2f(c_x, c_y)
        rect_size = Size2f(zone_size, zone_size)
        conf_data.roi = Rect(rect_center, rect_size)
        cfg = SpatialLocationCalculatorConfig()
        cfg.addROI(conf_data)
        node.io['spatial_location_config'].send(cfg)
        ${_TRACE2} ("Manager sent ROI to spatial_location_config")
        # Wait xyz response
        xyz_data = node.io['spatial_data'].get().getSpatialLocations()
        ${_TRACE2} ("Manager received spatial_location")
        coords = xyz_data[0].spatialCoordinates
        xyz = [coords.x, coords.y, coords.z]
        roi = xyz_data[0].config.roi
        xyz_zone = [int(roi.topLeft().x - crop_w), int(roi.topLeft().y), int(roi.bottomRight().x - crop_w), int(roi.bottomRight().y)]
        ${_IF_XYZ}

        # Send result to host
        send_result_hand(2 if send_new_frame_to_branch==1 else 0, nb_lm_inf, lm_score, handedness, sqn_rr_center_x, sqn_rr_center_y, sqn_rr_size, rotation, rrn_lms, sqn_lms, world_lms, xyz, xyz_zone)
        send_new_frame_to_branch = 2 

        # Calculate the ROI for next frame
        # Compute rotation
        x0 = sqn_lms[0]
        y0 = sqn_lms[1]
        x1 = 0.25 * (sqn_lms[2*id_index_mcp] + sqn_lms[2*id_ring_mcp]) + 0.5 * sqn_lms[2*id_middle_mcp]
        y1 = 0.25 * (sqn_lms[2*id_index_mcp+1] + sqn_lms[2*id_ring_mcp+1]) + 0.5 * sqn_lms[2*id_middle_mcp+1]
        rotation = 0.5 * pi - atan2(y0 - y1, x1 - x0)
        rotation = normalize_radians(rotation)
        # Find boundaries of landmarks
        min_x = min_y = 1
        max_x = max_y = 0
        for id in ids_for_bounding_box:
            min_x = min(min_x, sqn_lms[2*id])
            max_x = max(max_x, sqn_lms[2*id])
            min_y = min(min_y, sqn_lms[2*id+1])
            max_y = max(max_y, sqn_lms[2*id+1])
        axis_aligned_center_x = 0.5 * (max_x + min_x)
        axis_aligned_center_y = 0.5 * (max_y + min_y)
        cos_rot = cos(rotation)
        sin_rot = sin(rotation)
        # Find boundaries of rotated landmarks
        min_x = min_y = 1
        max_x = max_y = -1
        for id in ids_for_bounding_box:
            original_x = sqn_lms[2*id] - axis_aligned_center_x
            original_y = sqn_lms[2*id+1] - axis_aligned_center_y
            projected_x = original_x * cos_rot + original_y * sin_rot
            projected_y = -original_x * sin_rot + original_y * cos_rot
            min_x = min(min_x, projected_x)
            max_x = max(max_x, projected_x)
            min_y = min(min_y, projected_y)
            max_y = max(max_y, projected_y)
        projected_center_x = 0.5 * (max_x + min_x)
        projected_center_y = 0.5 * (max_y + min_y)
        center_x = (projected_center_x * cos_rot - projected_center_y * sin_rot + axis_aligned_center_x)
        center_y = (projected_center_x * sin_rot + projected_center_y * cos_rot + axis_aligned_center_y)
        width = (max_x - min_x)
        height = (max_y - min_y)
        #
        sqn_rr_size = 2 * max(width, height) 
        sqn_rr_center_x = (center_x + 0.1 * height * sin_rot) 
        sqn_rr_center_y = (center_y - 0.1 * height * cos_rot) 
        ${_TRACE1} (f"Landmarks - hand confirmed")
    else:
        send_result_no_hand(2 if send_new_frame_to_branch==1 else 0, nb_lm_inf)
        send_new_frame_to_branch = 0
        previous_hand_label = None
        ${_TRACE1} (f"Landmarks - hand not confirmed")