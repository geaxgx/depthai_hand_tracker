# This script has to be run from the docker container started by ./docker_tflite2tensorflow.sh
source /opt/intel/openvino_2021/bin/setupvars.sh

# Palm detection model
tflite2tensorflow \
  --model_path palm_detection.tflite \
  --model_output_path palm_detection \
  --flatc_path ../../flatc \
  --schema_path ../../schema.fbs \
  --output_pb True \
  --optimizing_for_openvino_and_myriad True
# Generate non normalized input models (openvino and blob): the normalization has to be mode explictly in the code
#tflite2tensorflow \
#  --model_path palm_detection.tflite \
#  --model_output_path palm_detection \
#  --flatc_path ../../flatc \
#  --schema_path ../../schema.fbs \
#  --output_openvino_and_myriad True 
# Generate normalized input models (openvino and blob)  for DepthAI
/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py --input_model palm_detection/model_float32.pb --model_name palm_detection --data_type FP16 --mean_values "[127.5, 127.5, 127.5]" --scale_values "[127.5, 127.5, 127.5]" --reverse_input_channels
/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/myriad_compile -m palm_detection.xml -ip u8 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4 -o palm_detection.blob

# Hand landmark model
tflite2tensorflow \
  --model_path hand_landmark.tflite \
  --model_output_path hand_landmark\
  --flatc_path ../../flatc \
  --schema_path ../../schema.fbs \
  --output_pb True
# Generate non normalized input models (openvino and blob): the normalization has to be mode explictly in the code
# tflite2tensorflow \
#   --model_path hand_landmark.tflite \
#   --model_output_path hand_landmark \
#   --flatc_path ../../flatc \
#   --schema_path ../../schema.fbs \
#   --output_openvino_and_myriad True
# Generate normalized input models for DepthAI
/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_tf.py --input_model hand_landmark/model_float32.pb --model_name hand_landmark --data_type FP16 --scale_values "[255.0, 255.0, 255.0]" --reverse_input_channels
/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/myriad_compile -m hand_landmark.xml -ip u8 -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4 -o hand_landmark.blob
