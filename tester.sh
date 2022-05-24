#!/bin/bash

out_path=(array_naive array_tiling direct_shared direct_global unroll_global unroll_cublass) # folder names created, output path for created txt files

# configuratin files in the format (C, HW, K)
C=(1 1 1 1 1 1 1 1 1 1 3 3 3 6 6 6 6 6 6 6 6 6 16 16 16 16 16 16 32 32) # 30
HW=(256 400 320 256 128 32 256 64 256 64 150 64 32 150 128 70 32 16 8 32 16 8 32 16 8 32 16 8 32 8) # 30
K=(3 6 6 6 6 6 9 9 12 12 16 16 16 16 16 16 16 16 16 32 32 32 32 32 32 64 64 64 64 64) # 30

#input file to change macro define
in_file=mbnet.h

# This block of code required to clear all macro defines
sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 0/' $in_file
sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 0/' $in_file
sed -i 's/define DIRECT .*/define DIRECT 0/' $in_file
sed -i 's/define CONV_SHARED .*/define CONV_SHARED 0/' $in_file
sed -i 's/define GEMM_GLOBAL .*/define GEMM_GLOBAL 0/' $in_file

#to change the nvprof files, please change the 'conf'
for j in ${!out_path[@]}; do
    
    if [[ ! -d ${out_path[$j]} ]] # Create folder if it does not exist
    then
        echo "File doesn't exist. Creating now"
        mkdir -p "${out_path[$j]}"
        echo "File created"
    else
        echo "File exists"
    fi
    
    case "$j" in # changes macro define values according to the method required
        #case 1
        0) sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 1/' ${in_file};;
        #case 2
        1) sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 1/' $in_file;;
        #case 3
        2) sed -i 's/define DIRECT .*/define DIRECT 1/' $in_file;;
        #case 4
        3) sed -i 's/define DIRECT .*/define DIRECT 1/' $in_file 
        sed -i 's/define CONV_SHARED .*/define CONV_SHARED 1/' $in_file;;
        #case 5
        4) sed -i 's/define GEMM_GLOBAL .*/define GEMM_GLOBAL 1/' $in_file;;
    esac

    

    for i in ${!C[@]}; do # loop to place all configuration files into use
        sed -i 's/define C .*/define C '${C[$i]}'/' ${in_file} # change C
        sed -i 's/define HW .*/define HW '${HW[$i]}'/' ${in_file} # change HW
        sed -i 's/define K .*/define K '${K[$i]}'/' ${in_file} # change K

        nvcc mbnet.cu -o mbnet -lcublas # compile it
        nvprof --log-file ${out_path[$j]}/nvprof_comp_${C[$i]}_${HW[$i]}_${K[$i]}.txt ./mbnet # stroe nvprof into the txt file
    done

    echo ' '

    # clears values after every iteration
    sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 0/' $in_file
    sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 0/' $in_file
    sed -i 's/define DIRECT .*/define DIRECT 0/' $in_file
    sed -i 's/define CONV_SHARED .*/define CONV_SHARED 0/' $in_file
    sed -i 's/define GEMM_GLOBAL .*/define GEMM_GLOBAL 0/' $in_file
done