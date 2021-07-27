import cv2
import numpy as np
from collections import namedtuple
from math import ceil, sqrt, exp, pi, floor, sin, cos, atan2, gcd


class HandRegion:
    """
        Attributes:
        pd_score : detection score
        pd_box : detection box [x, y, w, h], normalized [0,1] in the squared image
        pd_kps : detection keypoints coordinates [x, y], normalized [0,1] in the squared image
        rect_x_center, rect_y_center : center coordinates of the rotated bounding rectangle, normalized [0,1] in the squared image
        rect_w, rect_h : width and height of the rotated bounding rectangle, normalized in the squared image (may be > 1)
        rotation : rotation angle of rotated bounding rectangle with y-axis in radian
        rect_x_center_a, rect_y_center_a : center coordinates of the rotated bounding rectangle, in pixels in the squared image
        rect_w, rect_h : width and height of the rotated bounding rectangle, in pixels in the squared image
        rect_points : list of the 4 points coordinates of the rotated bounding rectangle, in pixels 
                expressed in the squared image during processing,
                expressed in the source rectangular image when returned to the user
        lm_score: global landmark score
        norm_landmarks : 3D landmarks coordinates in the rotated bounding rectangle, normalized [0,1]
        landmarks : 2D landmarks coordinates in pixel in the source rectangular image
        handedness: float between 0. and 1., > 0.5 for right hand, < 0.5 for left hand,
        label: "left" or "right", handedness translated in a string,
        xyz: real 3D world coordinates of the wrist landmark, or of the palm center (if landmarks are not used),
        xyz_zone: (left, top, right, bottom), pixel coordinates in the source rectangular image 
                of the rectangular zone used to estimate the depth
        """
    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        self.pd_score = pd_score # Palm detection score 
        self.pd_box = pd_box # Palm detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps # Palm detection keypoints

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))


SSDAnchorOptions = namedtuple('SSDAnchorOptions',[
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',
        'interpolated_scale_aspect_ratio',
        'fixed_anchor_size'])

def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    if num_strides == 1:
        return (min_scale + max_scale) / 2
    else:
        return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1)

def generate_anchors(options):
    """
    option : SSDAnchorOptions
    # https://github.com/google/mediapipe/blob/master/mediapipe/calculators/tflite/ssd_anchors_calculator.cc
    """
    anchors = []
    layer_id = 0
    n_strides = len(options.strides)
    while layer_id < n_strides:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []
        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while last_same_stride_layer < n_strides and \
                options.strides[last_same_stride_layer] == options.strides[layer_id]:
            scale = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer, n_strides)
            if last_same_stride_layer == 0 and options.reduce_boxes_in_lowest_layer:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios += [1.0, 2.0, 0.5]
                scales += [0.1, scale, scale]
            else:
                aspect_ratios += options.aspect_ratios
                scales += [scale] * len(options.aspect_ratios)
                if options.interpolated_scale_aspect_ratio > 0:
                    if last_same_stride_layer == n_strides -1:
                        scale_next = 1.0
                    else:
                        scale_next = calculate_scale(options.min_scale, options.max_scale, last_same_stride_layer+1, n_strides)
                    scales.append(sqrt(scale * scale_next))
                    aspect_ratios.append(options.interpolated_scale_aspect_ratio)
            last_same_stride_layer += 1
        
        for i,r in enumerate(aspect_ratios):
            ratio_sqrts = sqrt(r)
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options.strides[layer_id]
        feature_map_height = ceil(options.input_size_height / stride)
        feature_map_width = ceil(options.input_size_width / stride)

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options.anchor_offset_x) / feature_map_width
                    y_center = (y + options.anchor_offset_y) / feature_map_height
                    # new_anchor = Anchor(x_center=x_center, y_center=y_center)
                    if options.fixed_anchor_size:
                        new_anchor = [x_center, y_center, 1.0, 1.0]
                        # new_anchor.w = 1.0
                        # new_anchor.h = 1.0
                    else:
                        new_anchor = [x_center, y_center, anchor_width[anchor_id], anchor_height[anchor_id]]
                        # new_anchor.w = anchor_width[anchor_id]
                        # new_anchor.h = anchor_height[anchor_id]
                    anchors.append(new_anchor)
        
        layer_id = last_same_stride_layer
    return np.array(anchors)

def generate_handtracker_anchors():
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
    anchor_options = SSDAnchorOptions(num_layers=4, 
                            min_scale=0.1484375,
                            max_scale=0.75,
                            input_size_height=128,
                            input_size_width=128,
                            anchor_offset_x=0.5,
                            anchor_offset_y=0.5,
                            strides=[8, 16, 16, 16],
                            aspect_ratios= [1.0],
                            reduce_boxes_in_lowest_layer=False,
                            interpolated_scale_aspect_ratio=1.0,
                            fixed_anchor_size=True)
    return generate_anchors(anchor_options)

def decode_bboxes(score_thresh, scores, bboxes, anchors, best_only=False):
    """
    wi, hi : NN input shape
    mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
    # Decodes the detection tensors generated by the model, based on
    # the SSD anchors and the specification in the options, into a vector of
    # detections. Each detection describes a detected object.

    https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt :
    node {
        calculator: "TensorsToDetectionsCalculator"
        input_stream: "TENSORS:detection_tensors"
        input_side_packet: "ANCHORS:anchors"
        output_stream: "DETECTIONS:unfiltered_detections"
        options: {
            [mediapipe.TensorsToDetectionsCalculatorOptions.ext] {
            num_classes: 1
            num_boxes: 896
            num_coords: 18
            box_coord_offset: 0
            keypoint_coord_offset: 4
            num_keypoints: 7
            num_values_per_keypoint: 2
            sigmoid_score: true
            score_clipping_thresh: 100.0
            reverse_output_order: true

            x_scale: 128.0
            y_scale: 128.0
            h_scale: 128.0
            w_scale: 128.0
            min_score_thresh: 0.5
            }
        }
    }

    scores: shape = [number of anchors 896]
    bboxes: shape = [ number of anchors x 18], 18 = 4 (bounding box : (cx,cy,w,h) + 14 (7 palm keypoints)
    """
    regions = []
    scores = 1 / (1 + np.exp(-scores))
    if best_only:
        best_id = np.argmax(scores)
        if scores[best_id] < score_thresh: return regions
        det_scores = scores[best_id:best_id+1]
        det_bboxes2 = bboxes[best_id:best_id+1]
        det_anchors = anchors[best_id:best_id+1]
    else:
        detection_mask = scores > score_thresh
        det_scores = scores[detection_mask]
        if det_scores.size == 0: return regions
        det_bboxes2 = bboxes[detection_mask]
        det_anchors = anchors[detection_mask]

    scale = 128 # x_scale, y_scale, w_scale, h_scale

    # cx, cy, w, h = bboxes[i,:4]
    # cx = cx * anchor.w / wi + anchor.x_center 
    # cy = cy * anchor.h / hi + anchor.y_center
    # lx = lx * anchor.w / wi + anchor.x_center 
    # ly = ly * anchor.h / hi + anchor.y_center
    det_bboxes = det_bboxes2* np.tile(det_anchors[:,2:4], 9) / scale + np.tile(det_anchors[:,0:2],9)
    # w = w * anchor.w / wi (in the prvious line, we add anchor.x_center and anchor.y_center to w and h, we need to substract them now)
    # h = h * anchor.h / hi
    det_bboxes[:,2:4] = det_bboxes[:,2:4] - det_anchors[:,0:2]
    # box = [cx - w*0.5, cy - h*0.5, w, h]
    det_bboxes[:,0:2] = det_bboxes[:,0:2] - det_bboxes[:,3:4] * 0.5

    for i in range(det_bboxes.shape[0]):
        score = det_scores[i]
        box = det_bboxes[i,0:4]
        # Decoded detection boxes could have negative values for width/height due
        # to model prediction. Filter out those boxes
        if box[2] < 0 or box[3] < 0: continue
        kps = []
        # 0 : wrist
        # 1 : index finger joint
        # 2 : middle finger joint
        # 3 : ring finger joint
        # 4 : little finger joint
        # 5 : 
        # 6 : thumb joint
        # for j, name in enumerate(["0", "1", "2", "3", "4", "5", "6"]):
        #     kps[name] = det_bboxes[i,4+j*2:6+j*2]
        for kp in range(7):
            kps.append(det_bboxes[i,4+kp*2:6+kp*2])
        regions.append(HandRegion(float(score), box, kps))
    return regions

def non_max_suppression(regions, nms_thresh):

    # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
    # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
    # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
    # boxes = [r.box for r in regions]
    boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]        
    scores = [r.pd_score for r in regions]
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh)
    return [regions[i[0]] for i in indices]

def normalize_radians(angle):
    return angle - 2 * pi * floor((angle + pi) / (2 * pi))

def rot_vec(vec, rotation):
    vx, vy = vec
    return [vx * cos(rotation) - vy * sin(rotation), vx * sin(rotation) + vy * cos(rotation)]

def detections_to_rect(regions):
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
    # # Converts results of palm detection into a rectangle (normalized by image size)
    # # that encloses the palm and is rotated such that the line connecting center of
    # # the wrist and MCP of the middle finger is aligned with the Y-axis of the
    # # rectangle.
    # node {
    #   calculator: "DetectionsToRectsCalculator"
    #   input_stream: "DETECTION:detection"
    #   input_stream: "IMAGE_SIZE:image_size"
    #   output_stream: "NORM_RECT:raw_roi"
    #   options: {
    #     [mediapipe.DetectionsToRectsCalculatorOptions.ext] {
    #       rotation_vector_start_keypoint_index: 0  # Center of wrist.
    #       rotation_vector_end_keypoint_index: 2  # MCP of middle finger.
    #       rotation_vector_target_angle_degrees: 90
    #     }
    #   }
    
    target_angle = pi * 0.5 # 90 = pi/2
    for region in regions:
        
        region.rect_w = region.pd_box[2]
        region.rect_h = region.pd_box[3]
        region.rect_x_center = region.pd_box[0] + region.rect_w / 2
        region.rect_y_center = region.pd_box[1] + region.rect_h / 2

        x0, y0 = region.pd_kps[0] # wrist center
        x1, y1 = region.pd_kps[2] # middle finger
        rotation = target_angle - atan2(-(y1 - y0), x1 - x0)
        region.rotation = normalize_radians(rotation)
        
def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    points = []
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [[p0x,p0y], [p1x,p1y], [p2x,p2y], [p3x,p3y]]

def rect_transformation(regions, w, h):
    """
    w, h : image input shape
    """
    # https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/palm_detection_detection_to_roi.pbtxt
    # # Expands and shifts the rectangle that contains the palm so that it's likely
    # # to cover the entire hand.
    # node {
    # calculator: "RectTransformationCalculator"
    # input_stream: "NORM_RECT:raw_roi"
    # input_stream: "IMAGE_SIZE:image_size"
    # output_stream: "roi"
    # options: {
    #     [mediapipe.RectTransformationCalculatorOptions.ext] {
    #     scale_x: 2.6
    #     scale_y: 2.6
    #     shift_y: -0.5
    #     square_long: true
    #     }
    # }
    # IMHO 2.9 is better than 2.6. With 2.6, it may happen that finger tips stay outside of the bouding rotated rectangle
    scale_x = 2.9
    scale_y = 2.9
    shift_x = 0
    shift_y = -0.5
    for region in regions:
        width = region.rect_w
        height = region.rect_h
        rotation = region.rotation
        if rotation == 0:
            region.rect_x_center_a = (region.rect_x_center + width * shift_x) * w
            region.rect_y_center_a = (region.rect_y_center + height * shift_y) * h
        else:
            x_shift = (w * width * shift_x * cos(rotation) - h * height * shift_y * sin(rotation)) #/ w
            y_shift = (w * width * shift_x * sin(rotation) + h * height * shift_y * cos(rotation)) #/ h
            region.rect_x_center_a = region.rect_x_center*w + x_shift
            region.rect_y_center_a = region.rect_y_center*h + y_shift

        # square_long: true
        long_side = max(width * w, height * h)
        region.rect_w_a = long_side * scale_x
        region.rect_h_a = long_side * scale_y
        region.rect_points = rotated_rect_to_points(region.rect_x_center_a, region.rect_y_center_a, region.rect_w_a, region.rect_h_a, region.rotation)

def hand_landmarks_to_rect(hand):
    # Calculates the ROI for the next frame from the current hand landmarks
    id_wrist = 0
    id_index_mcp = 5
    id_middle_mcp = 9
    id_ring_mcp =13
    
    lms_xy =  hand.landmarks[:,:2]
    # print(lms_xy)
    # Compute rotation
    x0, y0 = lms_xy[id_wrist]
    x1, y1 = 0.25 * (lms_xy[id_index_mcp] + lms_xy[id_ring_mcp]) + 0.5 * lms_xy[id_middle_mcp]
    rotation = 0.5 * pi - atan2(y0 - y1, x1 - x0)
    rotation = normalize_radians(rotation)
    # Now we work only on a subset of the landmarks
    ids_for_bounding_box = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
    lms_xy = lms_xy[ids_for_bounding_box]
    # Find center of the boundaries of landmarks
    axis_aligned_center = 0.5 * (np.min(lms_xy, axis=0) + np.max(lms_xy, axis=0))
    # Find boundaries of rotated landmarks
    original = lms_xy - axis_aligned_center
    c, s = np.cos(rotation), np.sin(rotation)
    rot_mat = np.array(((c, -s), (s, c)))
    projected = original.dot(rot_mat)
    min_proj = np.min(projected, axis=0)
    max_proj = np.max(projected, axis=0)
    projected_center = 0.5 * (min_proj + max_proj)
    center = rot_mat.dot(projected_center) + axis_aligned_center
    width, height = max_proj - min_proj
    next_hand = HandRegion()
    next_hand.rect_w_a = next_hand.rect_h_a = 2 * max(width, height)
    next_hand.rect_x_center_a = center[0] + 0.1 * height * s
    next_hand.rect_y_center_a = center[1] - 0.1 * height * c
    next_hand.rotation = rotation
    next_hand.rect_points = rotated_rect_to_points(next_hand.rect_x_center_a, next_hand.rect_y_center_a, next_hand.rect_w_a, next_hand.rect_h_a, next_hand.rotation)
    return next_hand

def warp_rect_img(rect_points, img, w, h):
        src = np.array(rect_points[1:], dtype=np.float32) # rect_points[0] is left bottom point !
        dst = np.array([(0, 0), (h, 0), (h, w)], dtype=np.float32)
        mat = cv2.getAffineTransform(src, dst)
        return cv2.warpAffine(img, mat, (w, h))

def distance(a, b):
    """
    a, b: 2 points in 3D (x,y,z)
    """
    return np.linalg.norm(a-b)

def angle(a, b, c):
    # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    # a, b and c : points as np.array([x, y, z]) 
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def find_isp_scale_params(size, resolution, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    resolution: sensor resolution (width, height)
    is_height : boolean that indicates if the value 'size' represents the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288 (first compatible size > lm_input_size)
    if size < 288:
        size = 288

    width, height = resolution

    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = height 
        other = width
    else:
        reference = width 
        other = height
    size_candidates = {}
    for s in range(288,reference,16):
        f = gcd(reference, s)
        n = s//f
        d = reference//f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)
            
    # What is the candidate size closer to 'size' ?
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist: break
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]

