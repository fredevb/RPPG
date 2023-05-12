import numpy as np

def min_max_norm(x, new_min, new_max):
    x_min, x_max = x.min(), x.max()
    return (x-x_min)/(x_max-x_min)*(new_max-new_min)+new_min

def min_max_norm_0_1(x):
    return min_max_norm(x, 0, 1)