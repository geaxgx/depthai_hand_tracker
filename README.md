# Hand tracking with DepthAI

Running Google Mediapipe Hand Tracking models on [DepthAI](https://docs.luxonis.com/en/gen2/) hardware (OAK-1, OAK-D, ...)

For an OpenVINO version, please visit : [openvino_hand_tracker](https://github.com/geaxgx/openvino_hand_tracker)

![Demo](img/hand_tracker.gif)

## Multi mode vs Solo mode
* In **Multi mode**, there is no limit on the number of hands detected. The palm detection model runs on every frame. Then for each palm detected, the landmark model runs 
on a ROI computed from the palm keypoints and supposed to contain the whole hand.
* In **Solo mode**, one hand max is detected. The palm detection model is run only on the first frame or when no hand was detected in the previous frame. If one or more hands are detected by the palm model, the landmark model then runs on the ROI associated to the palm with the highest score. From the landmarks, the ROI of the next frame is computed based on the assimption that the hand does not move a lot between two consecutive frames. So because the palm detection runs only once in a while, **the FPS in Solo mode is significantly higher**. In contrast, with Multi mode, because you want to be able to detect any new appearing hand, you need to run the palm model on every frame. In addition, Solo mode is also **more robust**: the palm detector works on the whole image and may easily miss small hands when the person is far from the camera, whereas the ROI computed from previous frame in Solo mode directly gives a focused zone to work with. In summary, if your application can work with only one hand max at any given time, the Solo mode is clearly the mode to choose.

*Note: currently, Edge mode (see below) supports only Solo mode.*

## Host mode vs Edge mode
Two modes are available:
- **Host mode :** aside the neural networks that run on the device, almost all the processing apart from nerual networks is run on the host (the only processing done on the device is the letterboxing operation before the pose detection network when using the device camera as video source). **Use this mode when you want to detect all the hands in the image or when the image comes from an external input source (video file, image file, host webcam).**
- **Edge mode :** most of the processing (neural networks, post-processings, image manipulations, ) is run on the device thanks to the depthai scripting node feature. It works only with the device camera but is **definitely the best option when working with the internal camera in Solo mode** (significantly faster than in Host mode). The data exchanged between the host and the device is minimal: landmarks of the detected hand (<1.5kB/frame), and optionally the device video frame. Edge mode only supports Solo mode.

## Body Pre Focusing 
**Body Pre Focusing is an optional mechanism meant to help the hand detection when the person is far from the camera.**

![Distance person-camera = 5m](img/body_pre_focusing_5m.gif)

The palm detector is trained to detect hands that are less than 2 meters away from the camera. So if the person stands further away, his hands may not be detected, especially when padding is used to make the image square. To improve the detection, a body pose estimator can help to focus on a zone of the image that contains only the hands. So instead of the whole image, we feed the palm detector with the cropped image of the zone around the hands.

A natural body pose estimator could be Blazepose as it is the model used in the Mediapipe Holistic solution, but here, we chose [Movenet Single pose](https://github.com/geaxgx/depthai_movenet/tree/main/examples/hand_focusing) because of its simpler architecture (Blazepose would imply a more complex pipeline with 2 more neural networks running on the MyriadX).

Movenet gives the wrist keypoints, which are used as the center of the zones we are looking for. Several options are available, selected by the **body_pre_focusing** parameter (illustrated in the table below). 

By setting the **hands_up_only** option, we ask to take into consideration only the hands for which the wrist keypoint is above the elbow keypoint, meaning in practice that the hands are raised. Indeed, when we want to recognize hand gestures, the arm is generally folded and the hand up.

*In the video above, the person is at 5m from the camera. The big yellow square represents the smart cropping zone of Movenet. The small green square represents the focus zone on which the palm detector is run. In this example, hands_up_only=True, so this green square appears only when the hand is raised. Once a hand is detected, only the landmark model runs, as long as it can track the hand (the small yellow rotated square is drawn around the hand)*

**Recommendations:**
* Use Body Pre Focusing when the person can be at more than 1.5 meter from the camera. When you know that the person always stays at a closer distance, palm detection on the whole image works well enough and Body Pre Focusing would just add an unnecessary overload in the processing. However, note that in Solo mode, once a hand has been successfully detected, the body pose network, like the palm detection network, stays inactive on the following frames, as long as the hand landmark regressor can keep track of the hand.
* In the typical use case of gesture recognition, the person raises his hand to make the pose. You probably want to set **hands_up_only** to True, as it is a convenient way to avoid false positive recognition when the hand is not raised. In addition, in Solo mode, **body_pre_focusing=higher** allows to focus on the hand that is higher. Still in Solo mode, another benefit of Body Pre Focusing is that most of the time (see remark below), handedness can be determined directly from the body keypoint, which is much more reliable than the handedness inferred by the landmark model, especially when the body is far from the camera.
* When the person is far, **good image definition and ambient lighting have a big impact on the quality of the detection**. Using a higher resolution (`--resolution ultra`) may help.

**Remarks:**
* There is a small chance that the handedness is incorrect if the focus zone given by the Body Pre Focusing algorithm contains several hands, which happens when hands are close to each other. In that case, the hand with the higher score is selected.
* When **hands_up_only** is used, it means that on the first frame where the hand is detected, the hand is raised. Then, on the following frames, the hand is tracked by the landmark model, and since its position relative to the elbow is not checked anymore, the hand can be lowered without interrupting the tracking.  

|Arguments|Palm detection input |Hand tracker output|Remarks|
|:-:|:-:|:-:|-|
|*No BPF*|<img src="img/pd_input_no_bpf.jpg" alt="PD input: no BPF" width="128"/>|[<img src="img/output_no_bpf.jpg" alt="Output: no BPF" width="350"/>](img/output_no_bpf.jpg)|Because of the padding, hands get very small and palm detection gives a poor result (right hand not detected, left hand detection inaccurate)|
|*No BPF*<br>crop=True|<img src="img/pd_input_no_bpf_crop.jpg" alt="PD input: no BPF, crop" width="128"/>|[<img src="img/output_no_bpf_crop.jpg" alt="Output: no BPF, crop" width="200"/>](img/output_no_bpf_crop.jpg)|Cropping the image along the shortest side is an easy and inexpensive way to improve the detection, but at the condition the person stays in the center of the image|
|body_pre_focusing=group<br>hands_up_only=False|<img src="img/pd_input_bpf_group_all_hands.jpg" alt="PD input: bpf=group, all hands" width="128"/>|[<img src="img/output_bpf_group_all_hands.jpg" alt="Output: bpf=group, all hands" width="350"/>](img/output_bpf_group_all_hands.jpg)|BPF algorithm finds a zone that contains both hands, which are correctly detected|
|body_pre_focusing=group<br>hands_up_only=True|<img src="img/pd_input_bpf_right.jpg" alt="PD input: bpf=group, hands up only" width="128"/>|[<img src="img/output_bpf_group.jpg" alt="Output: bpf=group, all hands" width="350"/>](img/output_bpf_group.jpg)|With "hands_up_only" set to True, the left hand is not taken into consideration since the wrist keypoint is below the elbow keypoint|
|body_pre_focusing=right|<img src="img/pd_input_bpf_right.jpg" alt="PD input: bpf=right" width="128"/>|[<img src="img/output_bpf_group.jpg" alt="Output: bpf=right" width="350"/>](img/output_bpf_group.jpg)|The right hand is correctly detected, whatever the value of "hands_up_only"|
|body_pre_focusing=left<br>hands_up_only=False|<img src="img/pd_input_bpf_left_all_hands.jpg" alt="PD input: bpf=left, all hands" width="128"/>|[<img src="img/output_bpf_left_all_hands.jpg" alt="Output:  bpf=left, all hands" width="350"/>](img/output_bpf_left_all_hands.jpg)|The left hand is correctly detected|
|body_pre_focusing=left<br>hands_up_only=true|<img src="img/pd_input_no_bpf.jpg" alt="PD input: bpf=left, hands up only" width="128"/>|[<img src="img/output_bpf_left.jpg" alt="Output: bpf=left, hands up only" width="350"/>](img/output_bpf_left.jpg)|Because the left hand is not raised, it is not taken into consideration, so we fall back to the case where BPF is not used|
|body_pre_focusing=higher|<img src="img/pd_input_bpf_right.jpg" alt="PD input: bpf=higher" width="128"/>|[<img src="img/output_bpf_higher.jpg" alt="Output: bpf=higher" width="350"/>](img/output_bpf_higher.jpg)|Here, same result as for "body_pre_focusing=right",  whatever the value of "hands_up_only"|


## Install

Install the python packages (depthai, opencv, open3d) with the following command:

```
python3 -m pip install -r requirements.txt
```

*Note that the version of depthai specified in requirements.txt contains a fix necessary to use Body Pre Focusing (-bpf) and Spatial Location (-xyz) together.*

## Run

**Usage:**
```
-> ./demo.py -h
usage: demo.py [-h] [-e] [-i INPUT] [--pd_model PD_MODEL] [--no_lm]
               [--lm_model LM_MODEL] [-s] [-xyz] [-g] [-c] [-f INTERNAL_FPS]
               [-r {full,ultra}]
               [--internal_frame_height INTERNAL_FRAME_HEIGHT]
               [-bpf {right,left,group,higher}] [-ah] [-t] [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit
  -e, --edge            Use Edge mode (postprocessing runs on the device)

Tracker arguments:
  -i INPUT, --input INPUT
                        Path to video or image file to use as input (if not
                        specified, use OAK color camera)
  --pd_model PD_MODEL   Path to a blob file for palm detection model
  --no_lm               Only the palm detection model is run (no hand landmark
                        model)
  --lm_model LM_MODEL   Path to a blob file for landmark model
  -s, --solo            Detect one hand max. Default in solo mode.
  -xyz, --xyz           Enable spatial location measure of palm centers
  -g, --gesture         Enable gesture recognition
  -c, --crop            Center crop frames to a square shape
  -f INTERNAL_FPS, --internal_fps INTERNAL_FPS
                        Fps of internal color camera. Too high value lower NN
                        fps (default= depends on the model)
  -r {full,ultra}, --resolution {full,ultra}
                        Sensor resolution: 'full' (1920x1080) or 'ultra'
                        (3840x2160) (default=full)
  --internal_frame_height INTERNAL_FRAME_HEIGHT
                        Internal color camera frame height in pixels
  -bpf {right,left,group,higher}, --body_pre_focusing {right,left,group,higher}
                        Enable Body Pre Focusing
  -ah, --all_hands      In Body Pre Focusing mode, consider all hands (not
                        only the hands up)
  -t, --trace           Print some debug messages

Renderer arguments:
  -o OUTPUT, --output OUTPUT
                        Path to output video file
```

- To use the color camera as input in Host mode:

    ```./demo.py```

- To use the color camera as input in Edge mode (recommended for Solo mode):

    ```./demo.py -e```

- To use the color camera as input in Edge mode when you don't need to retrieve the video frame (only the landmark information is transfered to the host):

    ```./demo.py -e -i rgb_laconic```

- To use a file (video or image) as input (Host mode only):

    ```./demo.py -i filename```

- To enable gesture recognition:

    ```./demo.py [-e] -g```

![Gesture recognition](img/gestures.gif)

- Recommended options for gesture recognition when the person can move a few meters from the camera:

    ```./demo.py -e -g -bpf higher```

    or

    ```./demo.py -e -g -bpf higher --resolution ultra```   (a bit slower but better image definition)

- To measure hand spatial location in camera coordinate system (only for depth-capable device like OAK-D):

    ```./demo.py [-e] -xyz```

    ![Hands spatial location](img/hands_xyz.png)

    The measure is made on the wrist keypoint (or on the palm box center if '--no_lm' is used).

- To run only the palm detection model (without hand landmarks, Host mode only):

    ```./demo.py --no_lm```

    ![Palm detection](img/palm_detection.png)

    Of course, gesture recognition is not possible in this mode.


|Keypress|Function|
|-|-|
|*Esc*|Exit|
|*space*|Pause|
|1|Show/hide the palm bounding box (only in non solo mode)|
|2|Show/hide the palm detection keypoints (only in non solo mode)|
|3|Show/hide the rotated bounding box around the hand|
|4|Show/hide landmarks|
|5|Show/hide handedness|
|6|Show/hide scores|
|7|Show/hide recognized gestures (-g or --gesture)|
|8|Show/hide hand spatial location (-xyz)|
|9|Show/hide the zone used to measure the spatial location (-xyz)|
|f|Show/hide FPS|
|b|Show/hide body keypoints, smart cropping zone and focus zone if body pre focusing is used (only in Host mode)|


## Mediapipe models 
You can find the models *palm_detector.blob* and *hand_landmark.blob* under the 'models' directory, but below I describe how to get the files.

1) Clone this github repository in a local directory (DEST_DIR)
2) In DEST_DIR/models directory, download the source tflite models from Mediapipe:
* [Palm Detection model](https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection.tflite)
* [Hand Landmarks model](https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/hand_landmark.tflite)
3) Install the amazing [PINTO's tflite2tensorflow tool](https://github.com/PINTO0309/tflite2tensorflow). Use the docker installation which includes many packages including a recent version of Openvino.
3) From DEST_DIR, run the tflite2tensorflow container:  ```./docker_tflite2tensorflow.sh```
4) From the running container: 
```
cd resources/models
./convert_models.sh
```
The *convert_models.sh* converts the tflite models in tensorflow (.pb), then converts the pb file into Openvino IR format (.xml and .bin), and finally converts the IR files in MyriadX format (.blob). 

By default, the number of SHAVES associated with the blob files is 4. In case you want to generate new blobs with different number of shaves, you can use the script *gen_blob_shave.sh*:
```
# Example: to generate blobs for 6 shaves
./gen_blob_shave.sh -m pd -n 6   # will generate palm_detection_sh6.blob
./gen_blob_shave.sh -m lm -n 6   # will generate hand_landmark_sh6.blob
```



**Explanation about the Model Optimizer params :**
- The preview of the OAK-* color camera outputs BGR [0, 255] frames . The original tflite palm detection model is expecting RGB [-1, 1] frames. ```--reverse_input_channels``` converts BGR to RGB. ```--mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5]``` normalizes the frames between [-1, 1].
- The images which are fed to hand landmark model are built on the host in a format similar to the OAK-* cameras (BGR [0, 255]). The original hand landmark model is expecting RGB [0, 1] frames. Therefore, the following arguments are used ```--reverse_input_channels --scale_values [255.0, 255.0, 255.0]```

**Movenet models :**
The 'lightning' and 'thunder' Movenet models come from the repository [geaxgx/depthai_movenet](https://github.com/geaxgx/depthai_movenet/tree/main/models).

## Custom model

The `custom_models` directory contains the code to build the following custom *DetectionBestCandidate* model. This model processes the outputs of the pose detection network (a 1x896x1 tensor for the scores and a 1x896x18 for the regressors) and yields the regressor with the highest score.

The method used to build these models is well explained on the [rahulrav's blog](https://rahulrav.com/blog/depthai_camera.html).

**Blob models vs tflite models**
The palm detection blob does not exactly give the same results as the tflite version, because the tflite ResizeBilinear instruction is converted into IR Interpolate-1. Yet the difference is almost imperceptible thanks to the great help of PINTO (see [issue](https://github.com/PINTO0309/tflite2tensorflow/issues/4) ).

## Code

To facilitate reusability, the code is splitted in 2 classes:
-  **HandTracker**, which is responsible of computing the hand landmarks. The importation of this class depends on the mode:
```
# For Host mode:
from HandTracker import HandTracker
```
```
# For Edge mode:
from HandTrackerEdge import HandTracker
```
- **HandTrackerRenderer**, which is responsible of rendering the landmarks on the video frame. 

This way, you can replace the renderer from this repository and write and personalize your own renderer (for some projects, you may not even need a renderer).

The file ```demo.py``` is a representative example of how to use these classes:
```
from HandTrackerRenderer import HandTrackerRenderer
from HandTracker import HandTracker

# The argparse stuff has been removed to keep only the important code

tracker = HandTracker(
        input_src=args.input, 
        use_lm= not args.no_lm, 
        use_gesture=args.gesture,
        xyz=args.xyz,
        solo=args.solo,
        crop=args.crop,
        resolution=args.resolution,
        stats=True,
        trace=args.trace,
        **tracker_args
        )

renderer = HandTrackerRenderer(
        tracker=tracker,
        output=args.output)

while True:
    # Run hand tracker on next frame
    frame, hands = tracker.next_frame()
    if frame is None: break
    # Draw hands
    frame = renderer.draw(frame, hands)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
tracker.exit()
```

`hands` returned by `tracker.next_frame()` is a list of HandRegion.

For more information on:
- the arguments of the tracker, please refer to the docstring of class `HandTracker` or `HandTrackerEdge` in `HandTracker.py` or `HandTrackerEdge.py`;
- the attributes of an `HandRegion` element you can exploit in your program, please refer to the doctring of class `HandRegion` in `mediapipe_utils.py`.

## Examples

|||
|-|-|
|[Pseudo-3D visualization with Open3d + smoothing filtering](examples/3d_visualization)  |[<img src="examples/3d_visualization/medias/3d_visualization.gif" alt="3D visualization" width="200"/>](examples/3d_visualization)|
|[Remote control with hand poses](examples/remote_control) |[<img src="examples/remote_control/medias/toggle_light.gif" alt="3D visualization" width="200"/>](examples/remote_control)|

## Credits
* [Google Mediapipe](https://github.com/google/mediapipe)
* Katsuya Hyodo a.k.a [Pinto](https://github.com/PINTO0309), the Wizard of Model Conversion !