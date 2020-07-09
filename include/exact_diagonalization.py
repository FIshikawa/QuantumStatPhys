import numpy as np
import copy

class ExactDiagonalization:
    def __init__(self, N, hamiltonian):
        self.N = N
        self.D = np.zeros(np.power(2,N), dtype=np.complex128)
        self.P = np.zeros((np.power(2,N), np.power(2,N)), dtype=np.complex128)
        self.D, self.P = np.linalg.eigh(hamiltonian)
        self.size = np.power(2,N)
        self.D_min = np.min(self.D)

    def __call__(self,density_matrix,t):
        N = self.N
        propagator_left = \
                np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
        propagator_right = \
                np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
        D_temp = np.zeros(np.power(2,N), dtype=np.complex128)
        for i in range(self.size):
            D_temp[i] = np.exp(-1.0j * t * (self.D[i] - self.D_min))
        D_temp = np.diag(D_temp)
        D_temp_inv = np.conjugate(D_temp)
        propagator_left = \
                np.matmul(np.matmul(self.P, D_temp), 
                          np.conjugate(self.P.transpose()))
        propagator_right = \
                np.matmul(np.matmul(self.P, D_temp_inv), 
                          np.conjugate(self.P.transpose()))
        rho_temp = np.matmul(np.matmul(propagator_left, 
                             density_matrix.density_matrix), propagator_right) 
        density_matrix_temp = density_matrix.copy()
        density_matrix_temp.density_matrix = rho_temp / np.trace(rho_temp).real
        return density_matrix_temp
