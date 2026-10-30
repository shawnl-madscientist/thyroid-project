from utils import calculate_mean, calculate_min, calculate_max
expression_levels = [4.72, 2.31, 8.15, 1.09, 6.44]
print("Mean:", calculate_mean(expression_levels))
print("Min:", calculate_min(expression_levels))
print("Max:", calculate_max(expression_levels))
def calculate_std(values):
	mean = calculate_mean(values)
	variance = sum((x - mean) ** 2 for x in values)/ len(values)
	return variance ** 0.5

def calculate_zscore(values):
	mean = calculate_mean(values)
	std = calculate_std(values)
	return [(x - mean) / std for x in values]
from utils import calculate_mean, calculate_min, calculate_max, calculate_std, calculate_zscore
print ("Std:", calculate_std(expression_levels))
print("z-scores:", calculate_zscore(expression_levels)
    )