import numpy as np
import matplotlib.pyplot as plt
from support import LinearSolversSparse, load_matrix

def test_solver(matrix_file, tol_values):
    A = load_matrix(matrix_file)
    if A is None:
        return {}
    b = np.random.rand(A.shape[0])  # Vettore dei termini noti casuale
    results = {}
    iterazione = 1  # Inizializza la variabile iterazione

    for tol in tol_values:
        print(f"Entrato nel for di test solver, iterazione {iterazione}\n")
        print(f"INIZIO iterazione numero {iterazione}\n")

        print("Inizio Jacobi")
        solver = LinearSolversSparse(A, b, tol=tol)
        results[f'Jacobi_tol_{tol}'] = solver.jacobi()
        print("Fine Jacobi \n")

        print("Inizio Gradiente")
        solver = LinearSolversSparse(A, b, tol=tol)
        results[f'Gradient_tol_{tol}'] = solver.gradient()
        print("Fine Gradiente \n")

        print("Inizio Gradiente Coniugato")
        solver = LinearSolversSparse(A, b, tol=tol)
        results[f'Conjugate_Gradient_tol_{tol}'] = solver.conjugate_gradient()
        print("Fine Gradiente Coniugato \n")

        print("Inizio Gauss-Seidel")
        solver = LinearSolversSparse(A, b, tol=tol)
        results[f'Gauss_Seidel_tol_{tol}'] = solver.gauss_seidel()
        print("Fine Gauss-Seidel \n")

        print(f"FINE iterazione numero {iterazione}\n")
        iterazione += 1  # Incrementa la variabile iterazione

    return results

def plot_results(results, matrix_file):
    methods = ['Jacobi', 'Gauss_Seidel', 'Gradient', 'Conjugate_Gradient']
    tol_values = [1e-4, 1e-6, 1e-8, 1e-10]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    for method in methods:
        iterations = []
        execution_times = []
        for tol in tol_values:
            key = f'{method}_tol_{tol}'
            if key in results:
                solution, iters, exec_time = results[key]
                iterations.append(iters)
                execution_times.append(exec_time)
            else:
                iterations.append(np.nan)
                execution_times.append(np.nan)

        # Filtra i valori non positivi
        positive_tol_values = [tol for tol, iter in zip(tol_values, iterations) if iter > 0]
        positive_iterations = [iter for iter in iterations if iter > 0]
        positive_execution_times = [time for time, iter in zip(execution_times, iterations) if iter > 0]

        if positive_iterations:
            ax1.plot(positive_tol_values, positive_iterations, marker='o', label=method)
            ax2.plot(positive_tol_values, positive_execution_times, marker='o', label=method)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Tolerance')
    ax1.set_ylabel('Iterations')
    ax1.set_title(f'Iterations vs. Tolerance for {matrix_file}')
    if ax1.has_data():
        ax1.legend()
    ax1.grid(True)

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Tolerance')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title(f'Execution Time vs. Tolerance for {matrix_file}')
    if ax2.has_data():
        ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(f'{matrix_file}_convergence.png')
    plt.show()

if __name__ == "__main__":
    print("Inizio")
    matrix_files = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]
    tol_values = [1e-4, 1e-6, 1e-8, 1e-10]

    for matrix_file in matrix_files:
        print(f"Processing matrix file: {matrix_file}")
        results = test_solver(matrix_file, tol_values)
        if isinstance(results, dict):  # Assicurati che results sia un dizionario
            for key, value in results.items():
                if value:  # Check if value is not None
                    solution, iterations, exec_time = value
                    print(f"{key}: Solution found in {iterations} iterations and {exec_time:.2f} seconds")

        plot_results(results, matrix_file)
