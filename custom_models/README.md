# Generation of the Palm Detection Post Processing model 

*This README explains how the file custom_models/PDPostProcessing_top2_sh1.blob was generated.*

In Edge mode, we want to do the post processing of the Palm Detection model on the device. As the post processing includes computations on large arrays, the method chosen consists in implementaing these calculations in a Pytorch nn.Module, following the method described in the [rahulrav's blog](https://rahulrav.com/blog/depthai_camera.html).
The pytorch module is exported in ONNX format then converted into OpenVINO IR and finally into a blob file `PDPostProcessing_top2_sh1.blob`.

*Note that in Host mode, the same post processing is done on the host with numpy and OpenCV nms function.*

## Install

To generate the Post Processing model, some python packages are needed: torch, onnx, onnx-simplifier, onnx_graphsurgeon. They can be installed with the following command:

```
> cd custom_models
> python3 -m pip install -r requirements.txt
```

*Note the restriction on torch version in requirements.txt. A newer version of torch (1.10) generates an [error](https://github.com/geaxgx/depthai_hand_tracker/issues/8) when converting the ONNX model to Openvino IR (2021.4).*

## Build the ONNX model

The post processing model:
* takes the 2 outputs of the Palm Detection model (scores: 1x896x1 and detections: 1x896x18),
* arranges a bit the datas before applying to them the Non Maximum Suppression (NMS) algorithm and yields the 2 best detections (2 corresponds to the number of hands max we want to detect). 

The script `generate_postproc_onnx.py` defines the pytorch nn.Module *PDPostProcessing*, exports it to an ONNX file then, using onnx_graphsurgeon, patches the ONNX file in order to limit the number of detections by NMS to 2. `PDPostProcessing_top2_raw.onnx` is the name of the unpatched model, and `PDPostProcessing_top2.onnx` the name of the patched model.
```
# From custom_models directory
> python generate_postproc_onnx.py
``` 

## Convert to a blob file

Start the tflite2tensorflow docker container (here we just use the OpenVINO distribution of the container, not the PINTO's tools):
```
# From custom_models directory
> ../docker_tflite2tensorflow.sh
```

Then, from the container shell, `convert_model.sh` converts the ONNX patched model to OpenVINO IR format wihich is compiled into `PDPostProcessing_top2_sh1.blob`:
```
./convert_model.sh  # Add `-s #` if you want to specify a number of shaves other than 1
```



