import csv
import os

nvprof_paths = ['direct_shared',
                'unroll_cublass', 'tenssort']  # folder paths


# this function finds a name of the kernel
def takeName(possible_header):
    if possible_header == '[CUDA memcpy DtoH]':
        return 'DtoH'
    if possible_header == '[CUDA memcpy HtoD]':
        return 'HtoD'
    if possible_header == '[CUDA memset]':
        return 'memset'
    word = ''
    for ch in possible_header:
        if ch == ' ':  # only one word required
            word = ''
            continue
        if ch == '(' or ch == '<':  # in case we find parameters we stop
            break
        word += ch
    return word


# function looks for all unique column names
def findHeaders(data, headers, AreMetrics=False):
    toStart = 0

    if not AreMetrics:  # checks is it a metrics collection mode
        for elem in data:
            elem = elem.split()  # split lines to words

            if elem[0] == 'API':  # if we reach the API, we already looked GPU kernels
                break

            if elem[0] == 'GPU':  # after this kernel starts
                toStart = 1

            if toStart == 1:
                possibleHeader = ''
                # go from backward since the kernel name is the last columns
                for word in reversed(elem):
                    if word[0].isdigit():
                        break
                    possibleHeader = word + " " + possibleHeader
                # function retrieves only name without conf
                header = takeName(possibleHeader.strip())
                if header not in headers:  # add only if unique
                    headers.append(header)
    else:
        for elem in data:
            elem = elem.split()

            if elem[0] == 'Kernel:':
                possibleHeader = ''
                # go from backward since the kernel name is the last columns
                for word in reversed(elem):
                    if word[0].isdigit():
                        break
                    possibleHeader = word + " " + possibleHeader
                # function retrieves only name without conf
                header = takeName(possibleHeader.strip())
                if header not in headers:  # add only if unique
                    headers.append(header)
    return headers


for i in range(0, len(nvprof_paths)):  # for each kernel method
    method = nvprof_paths[i] + '_sum.csv'
    with open(method, 'w', newline='') as fopen:  # create csv file
        files = os.listdir(nvprof_paths[i])
        header = ['Configuration']  # first column to indicate configuration

        for file in files:  # for every file in folders
            with open(nvprof_paths[i]+'/'+file, 'r') as log:
                data = log.readlines()
                # function returns all found headers
                header = findHeaders(data, header)

        writer = csv.writer(fopen)
        header.append('Total_time')  # column for total time
        header.append('Kernel_time')  # column for toal time - HtoD - DtoH
        writer.writerow(header)
    print('Created csv file ' + method)

AreMetrics = False

metrics = ['sm_efficiency', 'achieved_occupancy', 'warp_execution_efficiency', 'inst_per_warp', 'gld_efficiency', 'gst_efficiency', 'shared_efficiency', 'shared_utilization',
           'l2_utilization', 'global_hit_rate', 'tex_cache_hit_rate', 'tex_utilization', 'ipc', 'inst_issued', 'inst_executed', 'issue_slot_utilization', 'dram_utilization']

for metric in metrics:  # we create csv file for every metrics

    out_file = 'metrics/'+metric+'_sum.csv'  # csv file created

    header = ['Configuration']  # column to store conf information

    with open(out_file, 'w', newline='') as fopen:  # open csv file to write
        for path in nvprof_paths:  # traverse each metrics txt file

            # we are interested only in this method, remove if there is more
            if path != 'direct_shared' and path != 'unroll_cublass':
                continue

            files = os.listdir('metrics/'+path)  # list directories
            for file in files:
                with open('metrics/'+path+'/'+file, 'r') as log:  # open each file
                    data = log.readlines()  # read their lines
                    header = findHeaders(data, header, metrics)
        writer = csv.writer(fopen)
        writer.writerow(header)
    print('Created csv file ' + out_file)
