import numpy as np
import statistics as stats

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]

mean_num = np.mean(data)
print("mean: ", mean_num)

mean_stats = stats.mean(data)
print("mean: ", mean_stats)

median_num = np.median(data)
print("Median: ", median_num)

median_stats = stats.median(data)
print("Median: ", median_stats)

mode_num = stats.mode(data)
print("Mode: ", mode_num)

range_num = np.ptp(data)
print("Range: ", range_num)

range_stats = max(data) - min(data)
print("Range: ", range_stats)

# quartiles
q1 = np.percentile(data, 25)
print("Q1: ", q1)

q2 = np.percentile(data, 50)
print("Q2: ", q2)

q3 = np.percentile(data, 75)
print("Q3: ", q3)

# Interquartile Range
iqr = q3 - q1
print("IQR: ", iqr)

std_dev_num = np.std(data)  
print("Standard Deviation: ", std_dev_num)

std_dev_stats = stats.stdev(data)
print("Standard Deviation: ", std_dev_stats)

variance_num = np.var(data)
print("Variance: ", variance_num)