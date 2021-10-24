# This script has to be run from the docker container started by ./docker_tflite2tensorflow.sh

usage ()
{
        echo "Generate a blob from an ONNX model with a specified number of shaves and cmx (nb cmx = nb shaves)"
        echo
        echo "Usage: ${0} [-m model_onnx] [-s nb_shaves]"
        echo
        echo "model: an existing .onnx file (default=PDPostProcessing_top2.onnx)"
        echo "nb_shaves: integer between 1 and 13 (default=1)"
}

while getopts ":hm:s:" opt; do
        case ${opt} in
                h )
                        usage
                        exit 0
                        ;;
                m )
                        model_onnx=$OPTARG
                        ;;
                s )
                        nb_shaves=$OPTARG
                        ;;
                : )
                        echo "Error: -$OPTARG requires an argument."
                        usage
                        exit 1
                        ;;
                \? )
                        echo "Invalid option: -$OPTARG"
                        usage
                        exit 1
                        ;;
        esac
done

if [ -z "$model_onnx" ]
then
       model_onnx=PDPostProcessing_top2.onnx
fi
if [ ! -f $model_onnx ]
then
        echo "The model ${model_onnx} does not exist"
        exit 1
fi
model=$(basename -s .onnx ${model_onnx})

if [ -z "$nb_shaves" ]
then
	nb_shaves=1
fi
if [ $nb_shaves -lt 1 -o $nb_shaves -gt 13 ]
then
        echo "Invalid number of shaves !"
        usage
        exit 1
fi

model_xml="${model}.xml"
model_blob="${model}_sh${nb_shaves}.blob"

echo Model: $model_xml $model_blob
echo Shaves: $nb_shaves

source /opt/intel/openvino_2021/bin/setupvars.sh

# Use FP16 and make the batch_size explicit.
python /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo_onnx.py \
                --input_model $model_onnx --data_type half 
/opt/intel/openvino_2021/deployment_tools/tools/compile_tool/compile_tool -d MYRIAD \
                -m $model_xml \
                -ip FP16 \
                -VPU_NUMBER_OF_SHAVES $nb_shaves \
                -VPU_NUMBER_OF_CMX_SLICES $nb_shaves \
                -o $model_blob
