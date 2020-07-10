import pytest
import sys
import os
include_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'include')
sys.path.append(include_file)
from physical_operators import *
import numpy as np
from density_matrix import DensityMatrix

@pytest.fixture(scope='module')
def set_density_matrix():
    N = 2
    pytest.density_matrix = DensityMatrix(N=N,relaxation=True)
    pytest.N = N
    yield set_density_matrix

def test_basic(set_density_matrix):
    N = pytest.N
    assert pytest.density_matrix.N == N
    assert 1 == pytest.density_matrix.trace_of_density_matrix()

def test_calc(set_density_matrix):
    N = pytest.N
    assert 0 == pytest.density_matrix.calculate_expectation_of_1body(O=Sz)
    assert 0 == pytest.density_matrix.calculate_expectation_of_1body(O=Sz,
                                                                    index=N//2)
    assert 0 == pytest.density_matrix.calculate_expectation_of_1body(O=Sz,
                                                                     index=0)
    assert 0 == pytest.density_matrix.calculate_expectation_of_2body(O1=Sz, 
                                                                     O2=Sz, 
                                                                     dist=1)

@pytest.fixture(scope='module')
def set_density_matrix_relaxation():
    N = 3
    pytest.density_matrix = DensityMatrix(N=N,tagged=N//2,relaxation=False)
    pytest.N = N
    yield set_density_matrix_relaxation

def test_basic_relaxation(set_density_matrix_relaxation):
    N = pytest.N
    assert pytest.density_matrix.N == N
    assert 1 == pytest.density_matrix.trace_of_density_matrix()

def test_calc_relaxation(set_density_matrix_relaxation):
    N = pytest.N
    assert 0 == pytest.density_matrix.calculate_expectation_of_1body(O=Sz)
    assert 1 == pytest.density_matrix.calculate_expectation_of_1body(O=Sx,
                                                                    index=N//2)
    assert 0 == pytest.density_matrix.calculate_expectation_of_1body(O=Sz,
                                                                     index=0)
    assert 0 == pytest.density_matrix.calculate_expectation_of_2body(O1=Sz, 
                                                                     O2=Sz, 
                                                                     dist=1)

def test_entanglement_entropy():
    whole_state = np.array([-1.0/3, 0, 2.0/3, 2.0/3],dtype=np.float64)
    rho = np.kron(np.reshape(whole_state,(4,-1)),whole_state)
    density_matrix = DensityMatrix(N=2,relaxation=True)
    density_matrix.density_matrix = rho.copy()
    reduced_density = density_matrix.reduced_density_matrix(left=0,right=0)
    answer_density = np.array([[1.0/9, -2.0/9], [-2.0/9, 8.0/9]],
                               dtype=np.float64)
    np.testing.assert_allclose(reduced_density,answer_density,atol=1e-07)
    entanglement_entropy = density_matrix.entanglement_entropy(left=0,right=0)
    D1 = (9 + np.sqrt(65))/18
    D2 = (9 - np.sqrt(65))/18
    answer_entropy = - D1 * np.log(D1) - D2 * np.log(D2) 
    np.testing.assert_almost_equal(entanglement_entropy,answer_entropy)

def test_system_entropy():
    N = 3
    rho = DensityMatrix(N=N,tagged=1,Sx_init=1.0,relaxation=False)
    rho_s = rho.system_density_matrix()
    print('rho_s : ')
    print(rho_s)
    rho_b = rho.bath_density_matrix()
    print('rho_b : ')
    print(rho_b)
    system_entropy = rho.system_entropy()
    bath_entropy = rho.bath_entropy()
    assert 0 == system_entropy
    assert (N-1)*np.log(2) == bath_entropy
    
