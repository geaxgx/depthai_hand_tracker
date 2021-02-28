# Hand tracking with DepthAI

Running Google Mediapipe Hand Tracking models on [DepthAI](https://docs.luxonis.com/en/gen2/) hardware (OAK-1, OAK-D, ...)

![Demo](img/hand_tracker.gif)
## Install

Install DepthAI gen2 python package:

```python3 -m pip install -r requirements.txt```

## Run

To use the color camera as input :

```python3 HandTracker.py```

To use a file (video or image) as input :

```python3 HandTracker.py -i filename```

To enable gesture recognition :

```python3 HandTracker.py -g```

![Gesture recognition](img/gestures.gif)

To run only the palm detection model (without hand landmarks), use *--no_lm* argument. Of course, gesture recognition is not possible in this mode.

Use keypress between 1 and 7 to enable/disable the display of hand features (palm bounding box, palm landmarks, hand landmarks, handedness, gesture,...)



## The models 
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
The *convert_models.sh* converts the tflite models in tensorflow (.pb), then converts the pb file into Openvino IR format (.xml and .bin), and finally converts the IR files in MyriadX format (.blob). You can modify this script, if for instance you want to change the number of shaves used by the models.



**Explanation about the Model Optimizer params :**
- The preview of the OAK-* color camera outputs BGR [0, 255] frames . The original tflite palm detection model is expecting RGB [-1, 1] frames. ```--reverse_input_channels``` converts BGR to RGB. ```--mean_values [127.5,127.5,127.5] --scale_values [127.5,127.5,127.5]``` normalizes the frames between [-1, 1].
- The images which are fed to hand landmark model are built on the host in a format similar to the OAK-* cameras (BGR [0, 255]). The original hand landmark model is expecting RGB [0, 1] frames. Therefore, the following arguments are used ```--reverse_input_channels --scale_values [255.0, 255.0, 255.0]```

**Blob models vs tflite models**
The blob palm detection model does not exactly give the same results as the tflite model, because the tflite ResizeBilinear instruction is converted into IR Interpolate-1. Yet the difference is almost imperceptible thanks to the great help of PINTO (see [issue](https://github.com/PINTO0309/tflite2tensorflow/issues/4) ).


