import torch
import torch.nn as nn
from torchvision.ops import nms
import numpy as np
import onnx
import sys, os
sys.path.insert(1, os.path.realpath(os.path.pardir))
from mediapipe_utils import generate_handtracker_anchors
import argparse


detection_input_length = 128
iou_threshold = 0.3
    
# In the comments below, N=896

class PDPostProcessing(nn.Module):
    def __init__(self, anchors, top_k):
        super(PDPostProcessing, self).__init__()
        self.top_k = top_k
        self.anchors = torch.from_numpy(anchors[:,:2]).float() # [N, 2]
        self.plus_anchor_center = np.array([[1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0], [0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1]])
        self.plus_anchor_center = torch.from_numpy(self.plus_anchor_center).float()

    def forward(self, x, y):
        # x.shape: [1, N, 1]
        # y.shape: [1, N, 18]

        scores = torch.sigmoid(x[0,:,0]) # [N]
        dets = torch.squeeze(y, 0) # [N, 18]
        # [N, 18] = [N, 18] + [N, 2] * [2, 18]
        dets = dets/detection_input_length + torch.mm(self.anchors, self.plus_anchor_center)

        # dets : the 4 first elements describe the square bounding box around the head (cx, cy, w, h)
        # cx,cy,w,h -> x1,y1,x2,y2 with:
        # x1 = cx - w/2
        # y1 = cy - h/2
        # x2 = cx + w/2
        # y2 = cy + h/2
        # bb_x1y1x2y2 = torch.mm(dets[:,:4],torch.from_numpy(np.array([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]])).float())
        bb_cxcy = dets[:,:2]
        bb_wh_half = dets[:,2:4] * 0.5
        bb_x1_y1 = bb_cxcy - bb_wh_half
        bb_x2_y2 = bb_cxcy + bb_wh_half
        bb_x1y1x2y2 = torch.cat((bb_x1_y1, bb_x2_y2), dim=1).float()


        # NMS
        # Parameters:
        # boxes: [N, 4] in (x1, y1, x2, y2) format
        # scores: [N]
        # iou_threshold: float
        # Returns: int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
        print(bb_x1y1x2y2.dtype, scores.dtype)
        keep_idx = nms(bb_x1y1x2y2, scores, iou_threshold)[:self.top_k]

        # The 14 elements of dets from 4 to 18 corresponds to 7 (x,y) normalized keypoints coordinates (useful for determining rotated rectangle)
        # Among the the 7 keypoints kps, we are interested only in kps[0] (wrist center) and kps[2] (middle finger)
        kp0 = dets[:,4:6][keep_idx]
        kp2 = dets[:,8:10][keep_idx]
        # sqn_rr_center_xy = kp0
        # sqn_scale_xy = kp2

        # We return (scores, cx, cy, w, kp0, kp2) (shape: [top_k, 8]) (no need of h since w=h)
        scores = torch.unsqueeze(scores[keep_idx], 1)
        cxcyw = dets[:,:3][keep_idx]
       
        dets = torch.cat((scores, cxcyw, kp0, kp2), dim=1)
        return dets


def test(anchors, top_k):

    model = PDPostProcessing(anchors, top_k)
    N = anchors.shape[0]
    X = torch.randn(1, N, 1, dtype=torch.float)
    Y = torch.randn(1, N, 18, dtype=torch.float)
    result = model(X, Y)
    print("Result shape:", result.shape)

def export_onnx(anchors, top_k, onnx_name):
    """
    Exports the model to an ONNX file.
    """
    model = PDPostProcessing(anchors, top_k)
    N = anchors.shape[0]
    X = torch.randn(1, N, 1, dtype=torch.float)
    Y = torch.randn(1, N, 18, dtype=torch.float)

    print(f"Generating {onnx_name}")
    torch.onnx.export(
        model,
        (X, Y),
        onnx_name,
        opset_version=11,
        do_constant_folding=True,
        # verbose=True,
        input_names=['classificators', 'regressors'],
        output_names=['result']
    )

def simplify(model):
    import onnxsim
    model_simp, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    print("Model has been simplified.")
    return model_simp

def patch_nms(model, top_k, score_thresh=None):
    import onnx_graphsurgeon as gs  
    import struct
    graph = gs.import_onnx(model)

    # Search for NMS node
    nms_not_found = True
    for node in graph.nodes:
        if node.op == "NonMaxSuppression":
            print("NonMaxSuppression found.")
            # Inputs of NonMaxSuppression:
            # 0: boxes
            # 1: scores
            # 2: max_output_boxes_per_class (opt)
            # 3: iou_threshold
            # 4: score_threshold

            # mobpc = max_out_boxes_per_class
            mobpc_input = node.inputs[2]
            # print(mobpc_input)
            # print(vars(mobpc_input))
            # print(mobpc_input._values)
            # print(vars(mobpc_input._values))
            mobpc = mobpc_input._values.tensor.raw_data
            mobpc = struct.unpack("q", mobpc)
            print("Current value of max_out_boxes_per_class", mobpc)
            new_mobpc = top_k
            mobpc = struct.pack("q", new_mobpc)
            mobpc_input._values.tensor.raw_data = mobpc
            print(f"max_out_boxes_per_class value changed to {top_k}")
            nms_not_found = False
            break
    assert nms_not_found==False, "NonMaxSuppression could not be found in the graph !"
    graph.cleanup().toposort()
    print("NonMaxSuppression has been patched")
    return gs.export_onnx(graph)

parser = argparse.ArgumentParser()
parser.add_argument('-top_k', type=int, default=2, help="max number of detections (default=%(default)i)")
parser.add_argument('-no_simp', action="store_true", help="do not run simplifier")
args = parser.parse_args()


top_k = args.top_k
run_simp = not args.no_simp

anchors = generate_handtracker_anchors().astype(float)
print(f"Nb anchors: {anchors.shape}", anchors.dtype) # [N, 4]

test(anchors, top_k)

name = f"PDPostProcessing_top{top_k}"
raw_onnx_name = f"{name}_raw.onnx"
export_onnx(anchors, args.top_k, raw_onnx_name)

model = onnx.load(raw_onnx_name)
print("Model IR version:", model.ir_version)
if run_simp:
    model = simplify(model)
model = patch_nms(model, top_k)
print("Model IR version:", model.ir_version)



onnx_name = f"{name}.onnx"
onnx.save(model, onnx_name)
print(f"Model saved in {onnx_name}")

