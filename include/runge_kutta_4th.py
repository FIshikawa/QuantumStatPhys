import numpy as np

class RungeKutta4th:
    def __init__(self, N, hamiltonian):
        self.N = N
        self.hamiltonian = hamiltonian

    def time_derivative(self,rho,t):
        return -1.0j * (np.matmul(self.hamiltonian(t), rho) - np.matmul(rho,self.hamiltonian(t).transpose()))

    def __call__(self,dt,density_matrix,t):
        h2 = dt / 2
        rho_temp1 = density_matrix.density_matrix.copy()
        rho_temp2 = density_matrix.density_matrix.copy()
        derivative1 = self.time_derivative(rho_temp1,t)
        rho_temp2 = rho_temp1 + h2 * derivative1 
        derivative2 = self.time_derivative(rho_temp2,t+h2)
        rho_temp2 = rho_temp1 + h2 * derivative2
        derivative3 = self.time_derivative(rho_temp2,t+h2)
        rho_temp2 = rho_temp1 + dt * derivative3
        derivative4 = self.time_derivative(rho_temp2,t+dt)
        rho_temp2 = rho_temp1 + dt / 6 * (derivative1 + 2 * derivative2 + 2 * derivative3 + derivative4)
        density_matrix.density_matrix = rho_temp2 / np.trace(rho_temp2).real
        return density_matrix
