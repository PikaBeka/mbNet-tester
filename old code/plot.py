import matplotlib.pyplot as plt
import numpy as np

methods = [
    'array_naive_sum.csv', 'array_tiling_sum.csv', 'pointer_naive_sum.csv', 'pointer_tiling_sum.csv']
labels = []
times = []

for method in methods:
    height = []
    with open(method, 'r') as csv_file:
        data = csv_file.readlines()
        for record in data[1:]:
            vals = record.split(',')
            if vals[0] not in labels:
                labels.append(vals[0])
            height.append(float(vals[4]))
        times.append(height)

w = 0.2

bar1 = np.arange(len(labels))

print(labels)
print(times[0])

for i in range(0, len(times)):
    if i == 0:
        bar = bar1
    else:
        bar = [i + w for i in bar]
    while len(times[i]) < len(bar):
        times[i].append(0.0)
    plt.bar(bar, times[i], w, label=methods[i])

plt.xlabel("Configurations (IF_N, INPUT_S, OF_N)")
plt.ylabel("Time(ms)")
plt.title("Execution time of different configurations")
plt.xticks(bar1, labels, rotation='vertical')
plt.legend()
plt.show()
