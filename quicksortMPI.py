from mpi4py import MPI
import numpy as np
import time

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return left, middle, right

def merge(left, middle, right):
    combined = np.concatenate((left, middle, right))
    return np.sort(combined)

def parallel_quicksort(arr):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_data = None
    if rank == 0:
        data_chunks = np.array_split(arr, size)
    else:
        data_chunks = None
    
    local_data = comm.scatter(data_chunks, root=0)

    left, middle, right = quicksort(local_data)

    # Recopilar y combinar los resultados
    sorted_data = comm.gather((left, middle, right), root=0)

    if rank == 0:
        combined_sorted = [np.concatenate([x[i] for x in sorted_data]) for i in range(3)]
        result = merge(*combined_sorted)
        return result

if __name__ == "__main__":
    data = np.array([86, 45, 43, 22, 1, 1, 2, 3, 20, 3, 41, 31, 78, 44, 23, 1, 97])

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Arreglo original:", data)

    start_time = time.time()

    sorted_data = parallel_quicksort(data)

    end_time = time.time()

    if rank == 0:
        print("Arreglo ordenado:", sorted_data.astype(int))
        print("Tiempo de ejecución:", end_time - start_time, "segundos")
