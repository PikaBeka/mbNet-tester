import matplotlib.pyplot as plt
import numpy as np

methods = ['direct_shared', 'unroll_cublass']
labels = []
times = []
isNormalized = False
AreMetrics = True

metrics = ['sm_efficiency', 'achieved_occupancy', 'warp_execution_efficiency', 'inst_per_warp', 'gld_efficiency', 'gst_efficiency', 'shared_efficiency', 'shared_utilization',
           'l2_utilization', 'global_hit_rate', 'tex_cache_hit_rate', 'tex_utilization', 'ipc', 'inst_issued', 'inst_executed', 'issue_slot_utilization', 'dram_utilization']


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


if not AreMetrics:
    for method in methods:
        height = []
        count = 0
        sum = 0
        with open(method + '_sum.csv', 'r') as csv_file:
            data = csv_file.readlines()  # read data
            for record in data[1:]:  # for every record
                vals = record.split(',')  # split records into vals
                if vals[0] not in labels:  # if label not added to list then add
                    labels.append(vals[0])
                count += 1
                sum += float(vals[-2])
                height.append(float(vals[-2]))  # append time
            height.append(sum / count)
            times.append(height)

    w = 1/(len(methods)+1)  # width of the bar

    labels.append('Average')

    bar = np.arange(len(labels))  # places of the labels

    if not isNormalized:

        for i in range(0, len(times)):
            # print(str(len(bar)) + " " + str(len(times[i])))
            # plots abar on x axis
            # print(times[i])
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
        #plt.setp(plt.gca(), ylim=(0, 7))

    plt.xlabel("Configurations (C, HW, K)")  # xlabel
    plt.title("Execution time of different configurations")  # title
    plt.xticks(bar + w * len(times)/2, labels, rotation='vertical')  # xtrics
    plt.legend()
    plt.show()

# else:
#     for metric in metrics:
#         height = []
#         count = 0
#         sum = 0
#         with open('metrics/'+metric+'_sum.csv', 'r') as input:
#             data = input.readlines()
#             for record in data[1:]:
#                 vals = record.split(',')
#                 if vals[0] not in labels:  # if label not added to list then add
#                     labels.append(vals[0])
#                 for i in range(1, len(vals)):
#                     if not isfloat(vals[i]):
#                         vals[i] = 0
#                     height.append(float(vals[i]))
#             times.append(height)

#     w = 1/(len(methods)+1)  # width of the bar
#     bar = np.arange(len(labels))
#     for i in range(0, len(times)):
#         print(str(len(bar)) + " " + str(len(times[i])))
#         # plots abar on x axis
#         # print(times)
#         plt.bar(bar + w * i, times[i], w, label=methods[i])
#     plt.ylabel("Time(ms)")  # ylabel
