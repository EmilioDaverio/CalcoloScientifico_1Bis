import numpy as np
import scipy
from scipy.io import mmread
from scipy.sparse import csr_matrix
from numpy.linalg import norm
import time

class LinearSolversSparse:
    def __init__(self, A, b, tol=1e-6, max_iter=20000):
        self.A = csr_matrix(A)
        self.b = b
        self.tol = tol
        self.max_iter = max_iter
        self.x = np.zeros_like(b)

    def jacobi(self):
        start_time = time.time()
        D_inv = 1.0 / self.A.diagonal()
        R = self.A - csr_matrix(np.diag(self.A.diagonal()))
        for k in range(self.max_iter):
            x_new = D_inv * (self.b - R.dot(self.x))
            if norm(self.A.dot(x_new) - self.b) / norm(self.b) < self.tol:
                end_time = time.time()
                print(f"Jacobi converged in {k+1} iterations and {end_time - start_time:.2f} seconds")
                return x_new, k + 1
            self.x = x_new
        end_time = time.time()
        print(f"Jacobi did not converge in {self.max_iter} iterations and {end_time - start_time:.2f} seconds")
        return self.x, self.max_iter

    def gauss_seidel(self):
        start_time = time.time()
        L = csr_matrix(np.tril(self.A.toarray()))
        U = self.A - L
        D = self.A.diagonal()
        for k in range(self.max_iter):
            x_new = np.copy(self.x)  # Copia l'iterazione corrente
            for i in range(self.A.shape[0]):
                sum1 = L[i, :i].dot(x_new[:i])
                sum2 = U[i, i+1:].dot(self.x[i+1:])
                x_new[i] = (self.b[i] - sum1 - sum2) / D[i]
            if norm(self.A.dot(x_new) - self.b) / norm(self.b) < self.tol:
                end_time = time.time()
                print(f"Gauss-Seidel converged in {k+1} iterations and {end_time - start_time:.2f} seconds")
                return x_new, k + 1
            self.x = x_new
        end_time = time.time()
        print(f"Gauss-Seidel did not converge in {self.max_iter} iterations and {end_time - start_time:.2f} seconds")
        return self.x, self.max_iter

    def gradient(self):
        start_time = time.time()
        r = self.b - self.A.dot(self.x)
        for k in range(self.max_iter):
            alpha = (r @ r) / (r @ self.A.dot(r))
            x_new = self.x + alpha * r
            if norm(self.A.dot(x_new) - self.b) / norm(self.b) < self.tol:
                end_time = time.time()
                print(f"Gradient Descent converged in {k+1} iterations and {end_time - start_time:.2f} seconds")
                return x_new, k + 1
            r = self.b - self.A.dot(x_new)
            self.x = x_new
        end_time = time.time()
        print(f"Gradient Descent did not converge in {self.max_iter} iterations and {end_time - start_time:.2f} seconds")
        return self.x, self.max_iter

    def conjugate_gradient(self):
        start_time = time.time()
        r = self.b - self.A.dot(self.x)
        p = r
        for k in range(self.max_iter):
            r_dot_r = r @ r
            alpha = r_dot_r / (p @ self.A.dot(p))
            x_new = self.x + alpha * p
            r_new = r - alpha * self.A.dot(p)
            if norm(self.A.dot(x_new) - self.b) / norm(self.b) < self.tol:
                end_time = time.time()
                print(f"Conjugate Gradient converged in {k+1} iterations and {end_time - start_time:.2f} seconds")
                return x_new, k + 1
            beta = (r_new @ r_new) / r_dot_r
            p = r_new + beta * p
            r = r_new
            self.x = x_new
        end_time = time.time()
        print(f"Conjugate Gradient did not converge in {self.max_iter} iterations and {end_time - start_time:.2f} seconds")
        return self.x, self.max_iter

def load_matrix(file_path):
    try:
        # Legge la matrice dal file e la converte in formato csr_matrix
        matrix = scipy.io.mmread(file_path)
        return matrix.tocsr()
    except ValueError as e:
        print(f"Error reading {file_path}: {e}")
        return None