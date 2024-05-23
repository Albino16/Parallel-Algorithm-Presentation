#!python /content/sample_data/
#! mpiexec --oversubscribe --allow-run-as-root -np 2 python /content/sample_data/
from mpi4py import MPI
import numpy as np

def selection_sort(arr):
    n = len(arr)
    for i in range(n-1):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

def parallel_selection_sort(arr):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Dividir la lista entre los procesos MPI
    local_n = len(arr) // size
    local_arr = np.empty(local_n, dtype=int)
    comm.Scatter(arr, local_arr, root=0)

    # Medir tiempo de inicio
    start_time = MPI.Wtime()

    # Ordenar localmente
    selection_sort(local_arr)

    # Medir tiempo de finalización
    end_time = MPI.Wtime()

    # Recopilar las partes ordenadas
    sorted_arr = None
    if rank == 0:
        sorted_arr = np.empty(len(arr), dtype=int)
    comm.Gather(local_arr, sorted_arr, root=0)

    # Calcular tiempo total de ejecución en el proceso raíz
    if rank == 0:
        total_time = end_time - start_time
        print("Tiempo de ejecución: {:.10f} segundos".format(total_time))

    return sorted_arr

if __name__ == "__main__":
    num_elements = 100  # Define la cantidad de elementos en el arreglo
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        data = np.random.randint(0, 100, num_elements)
        print("Arreglo original:", data)
    else:
        data = None

    sorted_data = parallel_selection_sort(data)

    if rank == 0:
        print("Arreglo ordenado:", sorted_data)