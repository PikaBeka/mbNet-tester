#!/bin/bash

file_s=naive.cu
file_t=tiling.cu

if_n=(1 1 1 1 1 1 1 1 1 3 3 3 6 6 6 6 6 6 6 6 6 16 16 16 16 16 16 32 32)
input_s=(400 320 256 256 256 256 128 64 64 32 150 64 32 150 128 70 32 32 16 16 8 8 32 32 16 16 8 8 32 8)
of_n=(6 6 3 6 9 12 6 9 12 6 16 16 16 16 16 16 16 32 16 32 16 32 32 64 32 64 32 64 64)

#to change the nvprof files, please change the 'conf'
conf=_naive_
conf_t=_tiling_

for i in ${!if_n[@]}; do
    sed -i 's/define IF_N .*/define IF_N '${if_n[$i]}'/' $file_s
    sed -i 's/define INPUT_S .*/define INPUT_S '${input_s[$i]}'/' $file_s
    sed -i 's/define OF_N .*/define OF_N '${of_n[$i]}'/' $file_s

    nvcc $file_s -o naive
    nvprof --log-file naive_profiler_results/nvprof_comp$conf${if_n[$i]}_${input_s[$i]}_${of_n[$i]}.txt ./naive

done

for i in ${!if_n[@]}; do
    sed -i 's/define IF_N .*/define IF_N '${if_n[$i]}'/' $file_t
    sed -i 's/define INPUT_S .*/define INPUT_S '${input_s[$i]}'/' $file_t
    sed -i 's/define OF_N .*/define OF_N '${of_n[$i]}'/' $file_t

    nvcc $file_t -o tiling
    nvprof --log-file tiling_profiler_results/nvprof_comp$conf_t${if_n[$i]}_${input_s[$i]}_${of_n[$i]}.txt ./tiling
done