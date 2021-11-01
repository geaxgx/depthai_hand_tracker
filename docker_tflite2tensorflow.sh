# Used to convert original Mediapipe tflite models to Openvino IR and MyriadX blob
# https://github.com/PINTO0309/tflite2tensorflow
#
# First run: docker pull ghcr.io/pinto0309/tflite2tensorflow:latest
docker run -it --rm \
    -v `pwd`:/home/user/workdir \
    ghcr.io/pinto0309/tflite2tensorflow:latest

