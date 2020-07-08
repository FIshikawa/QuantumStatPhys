import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'include'))

import copy
import numpy as np
from physical_operators import *
from exact_diagonalization import ExactDiagonalization

class TrialDensity:
    def __init__(self,N):
        if(N==2):
            rho_temp = np.kron(np.array([[1,0],[1,0]],dtype=np.complex128),np.eye(2,dtype=np.complex128))
        elif(N==1):
            rho_temp = np.eye(np.power(2,N),dtype=np.complex128)
        rho_temp = rho_temp / np.trace(rho_temp)
        self.density_matrix = rho_temp

    def trace_of_density_matrix(self):
        return np.trace(self.density_matrix)

    def total_energy(self,hamiltonian):
        return np.trace(np.dot(hamiltonian,self.density_matrix)).real 

    def copy(self):
        return copy.deepcopy(self)


@pytest.fixture(scope='module')
def set_matrix():
    # simple matrix h*Sz 
    h = np.float64(1.0)
    N = 1
    H_exact = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
    H_exact[0][0] =  h; H_exact[0][1] =  0; 
    H_exact[1][0] =  0; H_exact[1][1] = -h;
    pytest.H_exact = H_exact
    pytest.h = h
    pytest.N = N
    pytest.exact_diagoanlization = ExactDiagonalization(N, H_exact)
    yield set_matrix

def test_basic(set_matrix):
    h = pytest.h
    assert pytest.exact_diagoanlization.N == pytest.N
    np.testing.assert_allclose(pytest.exact_diagoanlization.D, np.array([-h,h],dtype=np.complex128)) 
    np.testing.assert_allclose(pytest.exact_diagoanlization.P, np.array([[0, 1],[1,0]])) 

def test_thermal_calc(set_matrix):
    dt = np.float64(0.01)
    h = pytest.h
    N = pytest.N
    test_density = TrialDensity(N)
    test_density = pytest.exact_diagoanlization(test_density,-0.5j)
    test_density = pytest.exact_diagoanlization(test_density,-0.5j)
    density_matrix_exact = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
    density_matrix_exact[0][0] =  np.exp(-2.0*h); density_matrix_exact[0][1] =  0; 
    density_matrix_exact[1][0] =  0; density_matrix_exact[1][1] = np.exp(2.0*h);
    density_matrix_exact /= np.trace(density_matrix_exact)
    np.testing.assert_allclose(test_density.density_matrix,density_matrix_exact,atol=1e-07)

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
    pytest.h = h
    pytest.N = N
    pytest.Sz_total = np.kron(Sz,np.eye(2,dtype=np.complex128)) + np.kron(np.eye(2,dtype=np.complex128),Sz)
    pytest.Sx_part = np.kron(Sx,np.eye(2,dtype=np.complex128))
    pytest.Sz_part = np.kron(Sz,np.eye(2,dtype=np.complex128))
    pytest.exact_diagoanlization = ExactDiagonalization(N, H_exact)
    yield set_hamiltonian

def test_real_time_develop_check_private_values(set_hamiltonian):
    h = pytest.h
    N = pytest.N
    init_D = pytest.exact_diagoanlization.D.copy()
    init_P = pytest.exact_diagoanlization.P.copy()
    assert pytest.exact_diagoanlization.N == pytest.N
    test_density = TrialDensity(N)
    test_density = pytest.exact_diagoanlization(test_density,t=100)
    np.testing.assert_allclose(pytest.exact_diagoanlization.D, init_D) 
    np.testing.assert_allclose(pytest.exact_diagoanlization.P, init_P) 

def test_real_time_develop_energy_conservation(set_hamiltonian):
    h = pytest.h
    N = pytest.N
    test_density = TrialDensity(N)
    for t in [10,100,1000]:
        E_init = test_density.total_energy(pytest.H_exact)
        Sz_total_init = np.trace(np.dot(pytest.Sz_total,test_density.density_matrix))
        Sx_part_init = np.trace(np.dot(pytest.Sx_part,test_density.density_matrix))
        Sz_part_init = np.trace(np.dot(pytest.Sz_part,test_density.density_matrix))
        test_density = pytest.exact_diagoanlization(TrialDensity(N),t)
        E_end = test_density.total_energy(pytest.H_exact)
        Sz_total_end = np.trace(np.dot(pytest.Sz_total,test_density.density_matrix))
        Sx_part_end = np.trace(np.dot(pytest.Sx_part,test_density.density_matrix))
        Sz_part_end = np.trace(np.dot(pytest.Sz_part,test_density.density_matrix))
        assert Sx_part_init != Sx_part_end
        assert Sz_part_init != Sz_part_end
        np.testing.assert_almost_equal(E_init,test_density.total_energy(pytest.H_exact),decimal=8)
        np.testing.assert_almost_equal(Sz_total_init,Sz_total_end,decimal=8)

