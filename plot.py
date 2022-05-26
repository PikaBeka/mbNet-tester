import matplotlib.pyplot as plt
import numpy as np
methods = ['direct_global', 'unroll_cublass']
labels = []
times = []
isNormalized = True

for method in methods:
    height = []
    with open(method + '_sum.csv', 'r') as csv_file:
        data = csv_file.readlines()  # read data
        for record in data[1:]:  # for every record
            vals = record.split(',')  # split records into vals
            if vals[0] not in labels:  # if label not added to list then add
                labels.append(vals[0])
            height.append(float(vals[-2]))  # append time
        times.append(height)

w = 1/(len(methods)+1)  # width of the bar

bar = np.arange(len(labels))  # places of the labels

if not isNormalized:

    for i in range(0, len(times)):
        # print(str(len(bar)) + " " + str(len(times[i])))
        # plots abar on x axis
        plt.bar(bar + w * i, times[i], w, label=methods[i])
    plt.ylabel("Time(ms)")  # ylabel

else:
    base = times[0].copy()

    for i in range(0, len(times)):
        for j in range(0, len(times[i])):
            # print(times[i][j])
            times[i][j] = base[j] / times[i][j]
        # print()
        plt.bar(bar + w * i, times[i], w, label=methods[i])

    plt.ylabel("Speedup")  # ylabel

plt.xlabel("Configurations (C, HW, K)")  # xlabel
plt.title("Execution time of different configurations")  # title
plt.xticks(bar + w * len(times)/2, labels, rotation='vertical')  # xtrics
plt.legend()
plt.show()
