#!/bin/bash

methods=(array/naive.cu array/tiling.cu pointer/naive.cu pointer/tiling.cu)
confs=(_array_naive_ _array_tiling_ _pointer_naive_ _pointer_tiling_)
exe=(array/naive array/tiling pointer/naive pointer/tiling)
path=(array/ array/ pointer/ pointer/)

# 1_256_3
# 1_400_6
# 1_320_6
# 1_256_6
# 1_128_6
# 1_32_6
# 1_256_9
# 1_64_9
# 1_256_12
# 1_64_12
# 3_150_16
# 3_64_16
# 3_32_16
# 6_150_16
# 6_128_16
# 6_70_16
# 6_32_16
# 6_16_16
# 6_8_16
# 6_32_32
# 6_16_32
# 6_8_32
# 16_32_32
# 16_16_32
# 16_8_32
# 16_32_64
# 16_16_64
# 16_8_64
# 32_32_64
# 32_8_64

if_n=(1 1 1 1 1 1 1 1 1 1 3 3 3 6 6 6 6 6 6 6 6 6 16 16 16 16 16 16 32 32) # 30
input_s=(256 400 320 256 128 32 256 64 256 64 150 64 32 150 128 70 32 16 8 32 16 8 32 16 8 32 16 8 32 8) # 30
of_n=(3 6 6 6 6 6 9 9 12 12 16 16 16 16 16 16 16 16 16 32 32 32 32 32 32 64 64 64 64 64)

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