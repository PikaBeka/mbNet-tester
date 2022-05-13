from cmath import log
import csv
import os
import re


class Parse:
    def __init__(self, log_file, if_n, input_s, of_n, o_file):
        self.log_file = log_file
        self.if_n = if_n
        self.input_s = input_s
        self.of_n = of_n
        self.sum = o_file

    def parse_time(self, timestamp):
        unit = timestamp[-1]
        time = timestamp[0:-1]
        if isinstance(time[-1], str):
            unit = time[-1] + unit
            time = time[0:-1]
        if unit == 's':
            time = float(time) * 1000
        if unit == 'us':
            time = round(float(time) * 0.001, 4)
        return time

    def parse_file(self):
        if not os.path.exists(self.log_file):
            print('Path to log file is invalid\n')
            return

        with open(self.sum, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            record = []
            with open(self.log_file, 'r') as log:
                data = log.readlines()[4].split()
                time_percentage = data[2][0:-1]
                time = self.parse_time(data[3])

            record.append(time_percentage)
            record.append(time)
            record.append(self.if_n)
            record.append(self.input_s)
            record.append(self.of_n)
            csv_writer.writerow(record)
            csv_file.close()


if __name__ == '__main__':
    if_n = [1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 16, 16, 16, 16, 16, 16, 32, 32]
    input_s = [400, 320, 256, 256, 256, 256, 128, 64, 64, 32, 150, 64,
               32, 150, 128, 70, 32, 32, 16, 16, 8, 8, 32, 32, 16, 16, 8, 8, 32, 8]
    of_n = [6, 6, 3, 6, 9, 12, 6, 9, 12, 6, 16, 16, 16, 16, 16,
            16, 16, 32, 16, 32, 16, 32, 32, 64, 32, 64, 32, 64, 64]

    methods = [
        'array_naive_sum.csv', 'array_tiling_sum.csv', 'pointer_naive_sum.csv', 'pointer_tiling_sum.csv']
    confs = ['array/_array_naive_', 'array/_array_tiling_',
             'pointer/_pointer_naive_', 'pointer/_pointer_tiling_', ]

    for j in range(0, len(methods)):
        for i in range(0, len(if_n)):
            if(j % 2 == 1 and len(if_n) - i == 2):
                break
            log_file = confs[j] + 'profiler_results/nvprof_comp' + \
                str(if_n[i]) + '_' + str(input_s[i]) + \
                '_' + str(of_n[i]) + '.txt'
            parser = Parse(log_file, int(if_n[i]), int(
                input_s[i]), int(of_n[i]), methods[j])
            parser.parse_file()
        print(methods[j] + " parsing finished")
