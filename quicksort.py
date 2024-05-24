import numpy as np
from numba import jit
import time

@jit(nopython=True)
def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

@jit(nopython=True)
def quicksort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)

@jit(nopython=True)
def parallel_quicksort(arr):
    quicksort(arr, 0, len(arr) - 1)

if __name__ == "__main__":
    #arr = np.array([8, 0, 4, 10, 12, -12, 14, -8, -9, 5, -9, -3, 0, 17, 19])
    arr = np.random.randint(low = 1, high=100, size=100000)
    start_time = time.time()
    parallel_quicksort(arr)
    end_time = time.time()
    
    print("Sorted array is:", arr)
    print("Execution time:", end_time - start_time, "seconds")
