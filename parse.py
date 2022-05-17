from cmath import log
import csv
import os


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
        if unit == 'ns':
            time = float(time) * 1e-6
        return time

    def takeName(self, possible_header):
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

    def takeRecord(self, elem, dict, total_time):
        # First need to find value
        # then find a name of the field
        possibleHeader = ''

        for word in reversed(elem):
            if word[0].isdigit():
                break
            possibleHeader = word + " " + possibleHeader

        header = self.takeName(possibleHeader)

        if elem[0] == 'GPU':
            time = self.parse_time(elem[3])
        else:
            time = self.parse_time(elem[1])

        dict[header] = time
        total_time += float(time)
        return total_time

    def parse_file(self):
        if not os.path.exists(self.log_file):
            print('Path to log file is invalid\n')
            return

        with open(self.sum, 'r') as csv_file:
            headers = csv_file.readlines()[0].split(',')

        with open(self.sum, 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
            record = {
                'Configuration': '(' + str(self.if_n) + '_' + str(self.input_s) + '_' + str(self.of_n) + ')'}
            with open(self.log_file, 'r') as log:
                data = log.readlines()
                toStart = 0
                total_time = 0.0
                for elem in data:
                    elem = elem.split()

                    if elem[0] == 'API':
                        break

                    if elem[0] == 'GPU':
                        toStart = 1

                    if toStart == 1:
                        total_time = self.takeRecord(elem, record, total_time)

            record['Total_time'] = "{:.3f}".format(total_time)
            record['Kernel_time\n'] = "{:.3f}".format(float(record['Total_time']) -
                                                      float(record['HtoD']) - float(record['DtoH']))
            csv_writer.writerow(record)
            csv_file.close()


if __name__ == '__main__':
    if_n = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 16, 16, 16, 16, 16, 16, 32, 32]
    input_s = [256, 400, 320, 256, 128, 32, 256, 64, 256, 64, 150, 64,
               32, 150, 128, 70, 32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 8]
    of_n = [3, 6, 6, 6, 6, 6, 9, 9, 12, 12, 16, 16, 16, 16, 16,
            16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64]

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
