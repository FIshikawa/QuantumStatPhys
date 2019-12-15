import numpy as np

class EulerMethod:
    def __init__(self, N, hamiltonian):
        self.N = N
        self.hamiltonian = hamiltonian

    def time_derivative(self,rho,t):
        return -1.0j * (np.matmul(self.hamiltonian(t), rho) - np.matmul(rho,self.hamiltonian(t).transpose()))

    def __call__(self,dt,density_matrix,t):
        rho_temp = density_matrix.density_matrix.copy()
        rho_temp = rho_temp + dt * self.time_derivative(rho_temp,t)
        density_matrix.density_matrix = rho_temp / np.trace(rho_temp).real
        return density_matrix
