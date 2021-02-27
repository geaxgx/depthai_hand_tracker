# Used to convert original Mediapipe tflite models to Openvino IR and MyriadX blob
# https://github.com/PINTO0309/tflite2tensorflow
#
# First run: docker pull pinto0309/tflite2tensorflow
docker run --gpus all -it --rm \
    -v `pwd`:/workspace/resources \
    -e LOCAL_UID=$(id -u $USER) \
    -e LOCAL_GID=$(id -g $USER) \
    pinto0309/tflite2tensorflow:latest bash 
