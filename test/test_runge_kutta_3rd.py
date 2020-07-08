import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'include'))

from physical_operators import *
import numpy as np
from runge_kutta_3rd import RungeKutta3rd

class TrialDensity:
    def __init__(self,N):
        rho_temp = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex64)
        rho_temp[0][0] = 1.0
        self.density_matrix = rho_temp

    def total_energy(self,hamiltonian):
        return np.trace(np.dot(hamiltonian,self.density_matrix)).real 


@pytest.fixture(scope='module')
def set_hamiltonian():
    # simple hamiltonian JSzSz + h Sx 
    h = 1.0
    J = 1.0
    N = 2
    H_exact = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex64)
    H_exact[0][0] =  J; H_exact[0][1] =  h; H_exact[0][2] =  h; H_exact[0][3] =  0; 
    H_exact[1][0] =  h; H_exact[1][1] = -J; H_exact[1][2] =  0; H_exact[1][3] =  h;
    H_exact[2][0] =  h; H_exact[2][1] =  0; H_exact[2][2] = -J; H_exact[2][3] =  h;
    H_exact[3][0] =  0; H_exact[3][1] =  h; H_exact[3][2] =  h; H_exact[3][3] =  J;
    pytest.H_exact = H_exact
    pytest.runge_kutta_3rd = RungeKutta3rd(N, hamiltonian=(lambda t : H_exact))
    pytest.h = h
    pytest.N = N
    yield set_hamiltonian

def test_basic(set_hamiltonian):
    h = pytest.h
    assert pytest.runge_kutta_3rd.N == pytest.N

def test_real_calc(set_hamiltonian):
    dt = np.float64(0.01)
    N_time = 1000
    h = pytest.h
    N = pytest.N
    test_density = TrialDensity(N)
    E_init = test_density.total_energy(pytest.H_exact)
    density_matrix_init = test_density.density_matrix.copy() 
    for i in range(N_time):
        test_density = pytest.runge_kutta_3rd(dt,test_density,t=0)
    assert (test_density.density_matrix != density_matrix_init).all()
    np.testing.assert_almost_equal(E_init,test_density.total_energy(pytest.H_exact),decimal=6)

