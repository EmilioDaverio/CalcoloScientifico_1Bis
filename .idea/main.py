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
        print(f"entrato nel for di test solver, iterazione {iterazione}\n")
        print(f"INIZIO iterazione numero {iterazione}\n")

        print("inizio jacobi")
        solver = LinearSolversSparse(A, b, tol=tol)
        results[f'Jacobi_tol_{tol}'] = solver.jacobi()
        print("fine jacobi \n")

        print("inizio gradiente")
        solver = LinearSolversSparse(A, b, tol=tol)
        results[f'Gradient_tol_{tol}'] = solver.gradient()
        print("fine gradiente \n")

        print("inizio gradiente 2")
        solver = LinearSolversSparse(A, b, tol=tol)
        results[f'Conjugate_Gradient_tol_{tol}'] = solver.conjugate_gradient()
        print("fine gradiente 2 \n")

        print("inizio gauss")
        solver = LinearSolversSparse(A, b, tol=tol)
        results[f'Gauss_Seidel_tol_{tol}'] = solver.gauss_seidel()
        print("fine gauss \n")

        print(f"FINE iterazione numero {iterazione}\n")
        iterazione += 1  # Incrementa la variabile iterazione

    return results

def plot_results(results, matrix_file):
    methods = ['Jacobi', 'Gauss_Seidel', 'Gradient', 'Conjugate_Gradient']
    tol_values = [1e-4, 1e-6, 1e-8, 1e-10]

    plt.figure(figsize=(10, 6))

    for method in methods:
        iterations = []
        for tol in tol_values:
            key = f'{method}tol{tol}'
            if key in results:
                solution, iters = results[key]
                iterations.append(iters)
            else:
                iterations.append(np.nan)

        # Filtra i valori non positivi
        positive_tol_values = [tol for tol, iter in zip(tol_values, iterations) if iter > 0]
        positive_iterations = [iter for iter in iterations if iter > 0]

        if positive_iterations:
            plt.plot(positive_tol_values, positive_iterations, marker='o', label=method)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Tolerance')
    plt.ylabel('Iterations')
    plt.title(f'Iterations vs. Tolerance for {matrix_file}')
    if plt.gca().has_data():
        plt.legend()
    plt.grid(True)
    plt.savefig(f'{matrix_file}_convergence.png')
    plt.show()

if __name__ == "__main__":
    print("inizio")
    matrix_files = ["spa1.mtx", "spa2.mtx", "vem1.mtx", "vem2.mtx"]
    tol_values = [1e-4, 1e-6, 1e-8, 1e-10]

    for matrix_file in matrix_files:
        print(f"Processing matrix file: {matrix_file}")
        results = test_solver(matrix_file, tol_values)
        if isinstance(results, dict):  # Assicurati che results sia un dizionario
            for key, value in results.items():
                if value:  # Check if value is not None
                    solution, iterations = value
                    print(f"{key}: Solution found in {iterations} iterations")

        plot_results(results, matrix_file)