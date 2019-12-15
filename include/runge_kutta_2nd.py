import numpy as np

class RungeKutta2nd:
    def __init__(self, N, hamiltonian):
        self.N = N
        self.hamiltonian = hamiltonian

    def time_derivative(self,rho,t):
        return -1.0j * (np.matmul(self.hamiltonian(t), rho) - np.matmul(rho,self.hamiltonian(t).transpose()))

    def __call__(self,dt,density_matrix,t):
        rho_temp1 = density_matrix.density_matrix.copy()
        rho_temp2 = density_matrix.density_matrix.copy()
        rho_temp2 = rho_temp1 + 0.5*dt * self.time_derivative(rho_temp1,t)
        rho_temp2 = rho_temp1 + dt * self.time_derivative(rho_temp2,t+0.5*dt)
        density_matrix.density_matrix = rho_temp2 / np.trace(rho_temp2).real
        return density_matrix
