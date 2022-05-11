import csv

naive_sum = 'naive_summary.csv'
tiling_sum = 'tiling_summary.csv'
with open(naive_sum, 'w', newline='') as fopen:
    writer = csv.writer(fopen)
    header = []
    header.append("Time(%)")
    header.append("Time(ms)")
    header.append("IF_N")
    header.append("INPUT_S")
    header.append("OF_N")
    writer.writerow(header)

with open(tiling_sum, 'w', newline='') as fopen:
    writer = csv.writer(fopen)
    header = []
    header.append("Time(%)")
    header.append("Time(ms)")
    header.append("IF_N")
    header.append("INPUT_S")
    header.append("OF_N")
    writer.writerow(header)
