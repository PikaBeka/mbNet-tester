#!/bin/bash

out_path=(array_naive array_tiling direct_shared unroll_cublass tenssort cudnn) # folder names created, output path for created txt files

# configuratin files in the format (C, HW, K)
C=(1 1 1 1 1 1 1 1 1 1 3 3 3 6 6 6 6 6 6 6 6 6 16 16 16 16 16 16 32 32) # 30
HW=(256 400 320 256 128 32 256 64 256 64 150 64 32 150 128 70 32 16 8 32 16 8 32 16 8 32 16 8 32 8) # 30
K=(3 6 6 6 6 6 9 9 12 12 16 16 16 16 16 16 16 16 16 32 32 32 32 32 32 64 64 64 64 64) # 30

#input file to change macro define
in_file=mbnet.h

is_metrics=false
is_trace=false
# is_jetson = /usr/local/cuda-10.2/bin/

# This block of code required to clear all macro defines
sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 0/' $in_file
sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 0/' $in_file
sed -i 's/define DIRECT .*/define DIRECT 0/' $in_file
sed -i 's/define CONV_SHARED .*/define CONV_SHARED 0/' $in_file
sed -i 's/define GEMM_GLOBAL .*/define GEMM_GLOBAL 0/' $in_file

mkdir -p "metrics"

while getopts m: flag
do
    case "${flag}" in
        m) method="${OPTARG}";;
    esac
done

if [ "${method}" != "array_naive" ] && [ "${method}" != "array_tiling" ] && [ "${method}" != "direct_shared" ] && [ "${method}" != "unroll_cublass" ] && [ "${method}" != "tensorrt" ] && [ "${method}" != "cudnn" ]; then
    echo "ERROR: Please supply one of the methods: array_naive, array_tiling, direct_shared, unroll_cubclass, tensorrt, cudnn"
    exit
fi

mkdir -p "${method}"
mkdir -p "metrics/${method}"
mkdir -p "trace/${method}"

if [[ "${method}" == "array_naive" ]]; then
	echo 'Running mbnet with array_naive method\n'
    sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 1/' ${in_file}
fi

if [[ "${method}" == "array_tiling" ]]; then
	echo 'Running mbnet with array_tiling method\n'
    sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 1/' $in_file
fi

if [[ "${method}" == "direct_shared" ]]; then
	echo 'Running mbnet with direct_shared method\n'
    sed -i 's/define DIRECT .*/define DIRECT 1/' $in_file
    sed -i 's/define CONV_SHARED .*/define CONV_SHARED 1/' $in_file
fi

if [[ "${method}" == "unroll_cublass" ]]; then
	echo 'Running mbnet with unroll_cublass method\n'
fi

if [[ "${method}" == "tensorrt" ]]; then
	echo 'Running mbnet with tenssort method\n'
fi

if [[ "${method}" == "cudnn" ]]; then
	echo 'Running mbnet with cudnn method\n'
    sed -i 's/define CUDNN .*/define CUDNN 1/' $in_file
fi

for i in ${!C[@]}; do # loop to place all configuration files into use
    if [[ "${method}" == "tensorrt" ]]; then
        sed -i 's/define C .*/define C '${C[$i]}'/' "S-LeNet-conv/${in_file}" # change C
        sed -i 's/define HW .*/define HW '${HW[$i]}'/' "S-LeNet-conv/${in_file}" # change HW
        sed -i 's/define K .*/define K '${K[$i]}'/' "S-LeNet-conv/${in_file}" # change K
        g++ -std=c++11 -o mbnet -I /usr/local/cuda-10.2/targets/aarch64-linux/include/ -I/usr/local/cuda-10.2/include -L/usr/local/cuda-10.2/targets/aarch64-linux/lib/ S-LeNet-conv/*.cpp S-LeNet-conv/*.cc -lnvinfer -lcuda -lcudart -lnvonnxparser -pthread -lprotobuf -lpthread -w  # compile it
    else
        sed -i 's/define C .*/define C '${C[$i]}'/' ${in_file} # change C
        sed -i 's/define HW .*/define HW '${HW[$i]}'/' ${in_file} # change HW
        sed -i 's/define K .*/define K '${K[$i]}'/' ${in_file} # change K
        /usr/local/cuda-10.2/bin/nvcc mbnet.cu -o mbnet -lcudart -lcublas -lcudnn -lstdc++ -lm # compile it
    fi

    if [[ "$is_metrics" = true ]]
    then
    #echo 'metrics run'
        /usr/local/cuda-10.2/bin/nvprof --aggregate-mode on --log-file metrics/${method}/nvprof_comp_${C[$i]}_${HW[$i]}_${K[$i]}.txt --metrics inst_issued ./mbnet #sm_efficiency,achieved_occupancy,warp_execution_efficiency,inst_per_warp,gld_efficiency,gst_efficiency,shared_efficiency,shared_utilization,l2_utilization,global_hit_rate,tex_cache_hit_rate,	tex_utilization,ipc,inst_issued,inst_executed,issue_slot_utilization,dram_utilization ./mbnet # stroe nvprof into the txt file
    else
        if [[ "$is_trace" = true ]]
        then
            /usr/local/cuda-10.2/bin/nvprof --log-file trace/${method}/nvprof_comp_${C[$i]}_${HW[$i]}_${K[$i]}.txt --print-gpu-trace ./mbnet
        else
            /usr/local/cuda-10.2/bin/nvprof --log-file ${method}/nvprof_comp_${C[$i]}_${HW[$i]}_${K[$i]}.txt ./mbnet # stroe nvprof into the txt file 
        fi
    fi
done
