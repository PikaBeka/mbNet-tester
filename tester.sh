#!/bin/bash

out_path=(array_naive array_tiling direct_shared unroll_cublass) # folder names created, output path for created txt files

# configuratin files in the format (C, HW, K)
C=(1 1 1 1 1 1 1 1 1 1 3 3 3 6 6 6 6 6 6 6 6 6 16 16 16 16 16 16 32 32) # 30
HW=(256 400 320 256 128 32 256 64 256 64 150 64 32 150 128 70 32 16 8 32 16 8 32 16 8 32 16 8 32 8) # 30
K=(3 6 6 6 6 6 9 9 12 12 16 16 16 16 16 16 16 16 16 32 32 32 32 32 32 64 64 64 64 64) # 30

#input file to change macro define
in_file=mbnet.h

is_metrics=true

# This block of code required to clear all macro defines
sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 0/' $in_file
sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 0/' $in_file
sed -i 's/define DIRECT .*/define DIRECT 0/' $in_file
sed -i 's/define CONV_SHARED .*/define CONV_SHARED 0/' $in_file
sed -i 's/define GEMM_GLOBAL .*/define GEMM_GLOBAL 0/' $in_file

mkdir -p "metrics"

# to change the nvprof files, please change the 'conf'
for j in ${!out_path[@]}; do
  
   mkdir -p "${out_path[$j]}"
   mkdir -p "metrics/${out_path[$j]}"
  
   if [[ $j -eq 0 ]]
   then
	echo 'skip array_naive'
	continue
       sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 1/' ${in_file}
   fi

   if [[ $j -eq 1 ]]
   then
	echo 'skip array_tiling'
	continue
       sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 1/' $in_file
   fi

   if [[ $j -eq 2 ]]
   then
       sed -i 's/define DIRECT .*/define DIRECT 1/' $in_file
       sed -i 's/define CONV_SHARED .*/define CONV_SHARED 1/' $in_file
   fi

   for i in ${!C[@]}; do # loop to place all configuration files into use
       sed -i 's/define C .*/define C '${C[$i]}'/' ${in_file} # change C
       sed -i 's/define HW .*/define HW '${HW[$i]}'/' ${in_file} # change HW
       sed -i 's/define K .*/define K '${K[$i]}'/' ${in_file} # change K

       nvcc mbnet.cu -o mbnet -lcublas # compile it
       if [[ "$is_metrics" = true ]]
       then
	   #echo 'metrics run'
           nvprof --log-file metrics/${out_path[$j]}/nvprof_comp_${C[$i]}_${HW[$i]}_${K[$i]}.txt --metrics sm_efficiency,achieved_occupancy,warp_execution_efficiency,inst_per_warp,gld_efficiency,gst_efficiency,shared_efficiency,shared_utilization,l2_utilization,global_hit_rate,tex_cache_hit_rate,tex_utilization,ipc,inst_issued,inst_executed,issue_slot_utilization,dram_utilization ./mbnet # stroe nvprof into the txt file
       else
           nvprof --log-file ${out_path[$j]}/nvprof_comp_${C[$i]}_${HW[$i]}_${K[$i]}.txt ./mbnet # stroe nvprof into the txt file 
           #echo "it wors"
       fi
   done

   echo ' '

   #clears values after every iteration
   sed -i 's/define ARRAY_NAIVE .*/define ARRAY_NAIVE 0/' $in_file
   sed -i 's/define ARRAY_TILING .*/define ARRAY_TILING 0/' $in_file
   sed -i 's/define DIRECT .*/define DIRECT 0/' $in_file
   sed -i 's/define CONV_SHARED .*/define CONV_SHARED 0/' $in_file
   sed -i 's/define GEMM_GLOBAL .*/define GEMM_GLOBAL 0/' $in_file
done

