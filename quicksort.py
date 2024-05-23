from mpi4py import MPI
import numpy as np
import time

# Función para ordenar un arreglo usando Quicksort
def quicksort(arreglo):
    if len(arreglo) <= 1:
        return arreglo
    pivote = arreglo[len(arreglo) // 2]
    izquierda, centro, derecha = [], [], []
    for x in arreglo:
        if x < pivote:
            izquierda.append(x)
        elif x == pivote:
            centro.append(x)
        else:
            derecha.append(x)
    #print("Partes ordenadas localmente:", izquierda, centro, derecha)
    return izquierda, centro, derecha

# Función para unir y ordenar las partes ordenadas de un arreglo
def merge(izquierda, centro, derecha):
    return np.sort(np.concatenate((izquierda, centro, derecha)))

# Función principal para realizar Quicksort paralelo
def parallel_quicksort(arreglo):
    # Inicializar comunicador MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Dividir el arreglo en partes y distribuirlo entre los procesos
    if rank == 0:
        data_chunks = np.array_split(arreglo, size)
    else:
        data_chunks = None
    
    local_data = comm.scatter(data_chunks, root=0)

    # Ordenar localmente cada parte del arreglo
    izquierda, centro, derecha = quicksort(local_data)

    # Recopilar y combinar los resultados ordenados
    sorted_data = comm.gather((izquierda, centro, derecha), root=0)

    # Combinar y ordenar los resultados si somos el proceso raíz
    if rank == 0:
        #print("Resultados combinados:", sorted_data)
        combined_sorted = [np.concatenate([x[i] for x in sorted_data]) for i in range(3)]
        #print("Combined sorted:", combined_sorted)
        if len(combined_sorted) == 3:  # Verificar si combined_sorted tiene la estructura correcta
            return merge(*combined_sorted)  # Llamar a merge solo si combined_sorted es correcto
        else:
            print("Error: combined_sorted no tiene la estructura correcta")
            return None
    else:
        return []

if __name__ == "__main__":
    num_elements = 100
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        data = np.random.randint(0, 100, num_elements)
        print("Original:", data)
    else:
        data = None

    start_time = time.time()
    sorted_data = parallel_quicksort(data)
    end_time = time.time()

    if rank == 0:
        print("Ordenado:", sorted_data.astype(int))
        print("Tiempo:", end_time - start_time, "segundos")
