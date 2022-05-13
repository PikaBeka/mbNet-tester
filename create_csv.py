import csv

methods = [
    'array_naive_sum.csv', 'array_tiling_sum.csv', 'pointer_naive_sum.csv', 'pointer_tiling._sum.csv']

for method in methods:
    with open(method, 'w', newline='') as fopen:
        writer = csv.writer(fopen)
        header = []
        header.append("Time(%)")
        header.append("Time(ms)")
        header.append("IF_N")
        header.append("INPUT_S")
        header.append("OF_N")
        writer.writerow(header)
