import csv
import os

nvprof_paths = ['array_naive', 'array_tiling',
                'direct_global', 'direct_shared', 'unroll_cublass', 'unroll_global']  # folder paths


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
def findHeaders(data, headers):
    toStart = 0
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
