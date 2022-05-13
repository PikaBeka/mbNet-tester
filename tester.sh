#!/bin/bash

methods=(array/naive.cu array/tiling.cu pointer/naive.cu pointer/tiling.cu)
confs=(_array_naive_ _array_tiling_ _pointer_naive_ _pointer_tiling_)
exe=(array/naive array/tiling pointer/naive pointer/tiling)
path=(array/ array/ pointer/ pointer/)

if_n=(1 1 1 1 1 1 1 1 1 3 3 3 6 6 6 6 6 6 6 6 6 16 16 16 16 16 16 32 32)
input_s=(400 320 256 256 256 256 128 64 64 32 150 64 32 150 128 70 32 32 16 16 8 8 32 32 16 16 8 8 32 8)
of_n=(6 6 3 6 9 12 6 9 12 6 16 16 16 16 16 16 16 32 16 32 16 32 32 64 32 64 32 64 64)

#to change the nvprof files, please change the 'conf'
for j in ${!methods[@]}; do
    for i in ${!if_n[@]}; do
        sed -i 's/define IF_N .*/define IF_N '${if_n[$i]}'/' ${methods[$j]}
        sed -i 's/define INPUT_S .*/define INPUT_S '${input_s[$i]}'/' ${methods[$j]}
        sed -i 's/define OF_N .*/define OF_N '${of_n[$i]}'/' ${methods[$j]}

        nvcc ${methods[$j]} -o ${exe[$j]}
        nvprof --log-file ${path[$j]}${confs[$j]}profiler_results/nvprof_comp${if_n[$i]}_${input_s[$i]}_${of_n[$i]}.txt ./${exe[$j]}

    done
done