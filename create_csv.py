import csv
import os

methods = [
    'array_naive_sum.csv', 'array_tiling_sum.csv', 'pointer_naive_sum.csv', 'pointer_tiling_sum.csv']

nvprof_paths = ['array/_array_naive_profiler_results', 'array/_array_tiling_profiler_results',
                'pointer/_pointer_naive_profiler_results', 'pointer/_pointer_tiling_profiler_results']


def takeName(possible_header):
    if possible_header == '[CUDA memcpy DtoH] ':
        return 'DtoH'
    if possible_header == '[CUDA memcpy HtoD] ':
        return 'HtoD'
    word = ''
    for ch in possible_header:
        if ch == '(':
            break
        word += ch
    return word


def findHeaders(data, headers):
    toStart = 0
    for elem in data:
        elem = elem.split()

        if elem[0] == 'API':
            break

        if elem[0] == 'GPU':
            toStart = 1

        if toStart == 1:
            possibleHeader = ''
            for word in reversed(elem):
                if word[0].isdigit():
                    break
                possibleHeader = word + " " + possibleHeader
            header = takeName(possibleHeader)
            if header not in headers:
                headers.append(header)
    return headers


for i in range(0, len(methods)):  # for each kernel method
    with open(methods[i], 'w', newline='') as fopen:  # create csv file
        files = os.listdir(nvprof_paths[i])
        header = ['Configuration']

        for file in files:
            with open(nvprof_paths[i]+'/'+file, 'r') as log:
                data = log.readlines()
                header = findHeaders(data, header)

        writer = csv.writer(fopen)
        header.append('Total_time')
        header.append('Kernel_time')
        writer.writerow(header)
    print('Created csv file ' + methods[i])
