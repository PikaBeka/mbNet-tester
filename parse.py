from cmath import log
import csv
import os
import create_csv as csv_file


# parsr class
class Parse:
    def __init__(self, log_file, C, HW, K, o_file):
        self.log_file = log_file
        self.C = C
        self.HW = HW
        self.K = K
        self.sum = o_file

    # function to work with time
    def parse_time(self, timestamp):
        unit = timestamp[-1]  # retreive first unit, mostly (s)
        time = timestamp[0:-1]  # remove the unit
        if isinstance(time[-1], str):  # if there is still unit like (m, n, u)
            unit = time[-1] + unit  # remove it and add to unit
            time = time[0:-1]
        # transform time to ms
        if unit == 's':
            time = float(time) * 1000
        if unit == 'us':
            time = round(float(time) * 0.001, 4)
        if unit == 'ns':
            time = float(time) * 1e-6
        return time

    def takeRecord(self, elem, dict, total_time):
        # First need to find value
        # then find a name of the field
        possibleHeader = ''

        for word in reversed(elem):
            if word[0].isdigit():
                break
            possibleHeader = word + " " + possibleHeader

        header = csv_file.takeName(
            possibleHeader.strip())  # look for kernel name

        if elem[0] == 'GPU':  # this case different since time will be in later columns
            time = self.parse_time(elem[3])
        else:
            time = self.parse_time(elem[1])  # obtains time

        dict[header] = time  # add it to header
        total_time += float(time)  # sum all times
        return total_time

    def parse_file(self):
        if not os.path.exists(self.log_file):
            print('Path to log file is invalid\n')
            return

        with open(self.sum, 'r') as csv_file:
            headers = csv_file.readlines()[0].split(',')  # get all headers

        with open(self.sum, 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=headers)
            record = {
                'Configuration': '(' + str(self.C) + '_' + str(self.HW) + '_' + str(self.K) + ')'}  # add first configuration value
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
                        # takes a record of the kernel time
                        total_time += self.takeRecord(elem, record, total_time)

            record['Total_time'] = "{:.3f}".format(total_time)
            record['Kernel_time\n'] = "{:.3f}".format(float(record['Total_time']) -
                                                      float(record['HtoD']) - float(record['DtoH']))
            csv_writer.writerow(record)
            csv_file.close()


if __name__ == '__main__':
    C = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 6, 6, 6,
         6, 6, 6, 6, 6, 6, 16, 16, 16, 16, 16, 16, 32, 32]
    HW = [256, 400, 320, 256, 128, 32, 256, 64, 256, 64, 150, 64,
          32, 150, 128, 70, 32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 16, 8, 32, 8]
    K = [3, 6, 6, 6, 6, 6, 9, 9, 12, 12, 16, 16, 16, 16, 16,
         16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64]

    for j in range(0, len(csv_file.nvprof_paths)):
        for i in range(0, len(C)):
            log_file = csv_file.nvprof_paths[j] + '/nvprof_comp_' + \
                str(C[i]) + '_' + str(HW[i]) + \
                '_' + str(K[i]) + '.txt'
            parser = Parse(log_file, int(C[i]), int(
                HW[i]), int(K[i]), csv_file.nvprof_paths[j]+'_sum.csv')
            parser.parse_file()
        print(csv_file.nvprof_paths[j] + " parsing finished")
