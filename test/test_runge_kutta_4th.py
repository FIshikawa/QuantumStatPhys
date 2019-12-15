import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'include'))

from physical_operators import *
import numpy as np
from runge_kutta_4th import RungeKutta4th

class TestDensity:
    def __init__(self,N):
        rho_temp = np.kron(np.array([[1,0],[1,0]],dtype=np.complex128),np.array([[0,1],[0,1]],dtype=np.complex128))
        rho_temp = rho_temp / np.trace(rho_temp)
        self.density_matrix = rho_temp

    def total_energy(self,hamiltonian):
        return np.trace(np.dot(hamiltonian,self.density_matrix)).real 


@pytest.fixture(scope='module')
def set_hamiltonian():
    # simple hamiltonian JSxSx + JSySy + h Sz 
    # in this system, total Sz is conserved
    h = 1.0
    J = 1.0
    N = 2
    H_exact = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
    H_exact[0][0] =2*h; H_exact[0][1] =  0; H_exact[0][2] =  0; H_exact[0][3] =  0; 
    H_exact[1][0] =  0; H_exact[1][1] =  0; H_exact[1][2] =2*J; H_exact[1][3] =  0;
    H_exact[2][0] =  0; H_exact[2][1] =2*J; H_exact[2][2] =  0; H_exact[2][3] =  0;
    H_exact[3][0] =  0; H_exact[3][1] =  0; H_exact[3][2] =  0; H_exact[3][3] =-2*h;
    pytest.H_exact = H_exact
    pytest.integrator = RungeKutta4th(N, hamiltonian=(lambda t : H_exact))
    pytest.h = h
    pytest.N = N
    pytest.Sz_total = np.kron(Sz,np.eye(2,dtype=np.complex128)) + np.kron(np.eye(2,dtype=np.complex128),Sz)
    pytest.Sx_part = np.kron(Sx,np.eye(2,dtype=np.complex128))
    pytest.Sz_part = np.kron(Sz,np.eye(2,dtype=np.complex128))
    yield set_hamiltonian

def test_basic(set_hamiltonian):
    h = pytest.h
    assert pytest.integrator.N == pytest.N

def test_real_calc(set_hamiltonian):
    dt = np.float64(0.01)
    N_time = 1000
    h = pytest.h
    N = pytest.N
    test_density = TestDensity(N)
    E_init = test_density.total_energy(pytest.H_exact)
    Sz_total_init = np.trace(np.dot(pytest.Sz_total,test_density.density_matrix))
    Sx_part_init = np.trace(np.dot(pytest.Sx_part,test_density.density_matrix))
    Sz_part_init = np.trace(np.dot(pytest.Sz_part,test_density.density_matrix))
    density_matrix_init = test_density.density_matrix.copy() 
    for i in range(N_time):
        test_density = pytest.integrator(dt,test_density,t=0)
    Sz_total_end = np.trace(np.dot(pytest.Sz_total,test_density.density_matrix))
    Sx_part_end = np.trace(np.dot(pytest.Sx_part,test_density.density_matrix))
    Sz_part_end = np.trace(np.dot(pytest.Sz_part,test_density.density_matrix))
    assert Sx_part_init != Sx_part_end
    assert Sz_part_init != Sz_part_end
    print(E_init)
    E_end = test_density.total_energy(pytest.H_exact)
    print(E_end)
    np.testing.assert_almost_equal(E_init, E_end)
    np.testing.assert_almost_equal(Sz_total_init,Sz_total_end)
