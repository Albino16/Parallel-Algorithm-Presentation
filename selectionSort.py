import numpy as np
from numba import jit
import time

@jit(nopython=True)
def selection_sort(a):
    n = len(a)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if a[j] < a[min_idx]:
                min_idx = j
        if min_idx != i:
            a[i], a[min_idx] = a[min_idx], a[i]
    return a

if __name__ == "__main__":
    arr = np.random.randint(low=1, high=100, size=100000)
    start_time = time.time()
    sorted_arr = selection_sort(arr)
    end_time = time.time()
    
    print("Sorted array is:", sorted_arr)
    print("Execution time:", end_time - start_time, "seconds")
#!python /content/sample_data/SelectionSort.py