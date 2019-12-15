import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'include'))

from physical_operators import *
import numpy as np
from hamiltonian import Hamiltonian

def test_time_independ():
    # simple hamiltonian JSzSz + h Sx 
    h = 0.5
    J = 1.0
    N = 2
    hamiltonian = Hamiltonian(N=N,tagged=-1)
    hamiltonian.add_H_1body(Sx, coef=h)
    hamiltonian.add_H_2body(Sz, Sz, coef=J)

    H_exact = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
    H_exact[0][0] =  J; H_exact[0][1] =  h; H_exact[0][2] =  h; H_exact[0][3] =  0; 
    H_exact[1][0] =  h; H_exact[1][1] = -J; H_exact[1][2] =  0; H_exact[1][3] =  h;
    H_exact[2][0] =  h; H_exact[2][1] =  0; H_exact[2][2] = -J; H_exact[2][3] =  h;
    H_exact[3][0] =  0; H_exact[3][1] =  h; H_exact[3][2] =  h; H_exact[3][3] =  J;

    assert hamiltonian.H_kind == 2
    assert hamiltonian.H_time_independ is None
    np.testing.assert_array_equal(hamiltonian(), H_exact)

def test_time_depend():
    # simple hamiltonian JSzSz + h Sx 
    J = 1.0
    N = 2
    hamiltonian = Hamiltonian(N=N,tagged=-1)
    hamiltonian.add_H_1body(Sx, coef=np.cos, time_depend=True)
    hamiltonian.add_H_2body(Sz, Sz, coef=J)
    
    t_test_fist = 2.0 * np.pi
    h = np.cos(t_test_fist)

    H_exact = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
    H_exact[0][0] =  J; H_exact[0][1] =  h; H_exact[0][2] =  h; H_exact[0][3] =  0; 
    H_exact[1][0] =  h; H_exact[1][1] = -J; H_exact[1][2] =  0; H_exact[1][3] =  h;
    H_exact[2][0] =  h; H_exact[2][1] =  0; H_exact[2][2] = -J; H_exact[2][3] =  h;
    H_exact[3][0] =  0; H_exact[3][1] =  h; H_exact[3][2] =  h; H_exact[3][3] =  J;

    assert hamiltonian.H_time_depend_kind == 1
    assert hamiltonian.H_kind == 1
    assert hamiltonian.time_depend is True
    np.testing.assert_array_equal(hamiltonian(t_test_fist), H_exact)

    t_test_second = np.pi
    h = np.cos(t_test_second)

    H_exact = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
    H_exact[0][0] =  J; H_exact[0][1] =  h; H_exact[0][2] =  h; H_exact[0][3] =  0; 
    H_exact[1][0] =  h; H_exact[1][1] = -J; H_exact[1][2] =  0; H_exact[1][3] =  h;
    H_exact[2][0] =  h; H_exact[2][1] =  0; H_exact[2][2] = -J; H_exact[2][3] =  h;
    H_exact[3][0] =  0; H_exact[3][1] =  h; H_exact[3][2] =  h; H_exact[3][3] =  J;
    np.testing.assert_array_equal(hamiltonian(t_test_second), H_exact)

def test_set_each_hamiltonian():
    N = 2
    h = 2.0
    Sz_on_tagged_only = Hamiltonian(N).operator_reshape(position=0,body=1,H_each=h*Sz)
    Sz_answer = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
    Sz_answer[0][0] =  h; Sz_answer[0][1] =  0; Sz_answer[0][2] =  0; Sz_answer[0][3] =  0; 
    Sz_answer[1][0] =  0; Sz_answer[1][1] = -h; Sz_answer[1][2] =  0; Sz_answer[1][3] =  0;
    Sz_answer[2][0] =  0; Sz_answer[2][1] =  0; Sz_answer[2][2] =  h; Sz_answer[2][3] =  0;
    Sz_answer[3][0] =  0; Sz_answer[3][1] =  0; Sz_answer[3][2] =  0; Sz_answer[3][3] = -h;

    np.testing.assert_array_equal(Sz_on_tagged_only, Sz_answer)

def test_just_tagged_hamiltonian():
    N = 2
    J = 1.0
    hamiltonian = Hamiltonian(N=N,tagged=0)
    hamiltonian.add_H_1body(Sz, coef=np.cos, time_depend=True, position=0)
    hamiltonian.add_H_1body(Sx, coef=J)

    t_test_fist = 2.0 * np.pi
    h = np.cos(t_test_fist)

    Sz_answer = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
    Sz_answer[0][0] =  h; Sz_answer[0][1] =  0; Sz_answer[0][2] =  J; Sz_answer[0][3] =  0; 
    Sz_answer[1][0] =  0; Sz_answer[1][1] = -h; Sz_answer[1][2] =  0; Sz_answer[1][3] =  J;
    Sz_answer[2][0] =  J; Sz_answer[2][1] =  0; Sz_answer[2][2] =  h; Sz_answer[2][3] =  0;
    Sz_answer[3][0] =  0; Sz_answer[3][1] =  J; Sz_answer[3][2] =  0; Sz_answer[3][3] = -h;

    np.testing.assert_array_equal(hamiltonian(t_test_fist), Sz_answer)

    t_test_second = np.pi
    h = np.cos(t_test_second)

    Sz_answer[0][0] =  h; Sz_answer[0][1] =  0; Sz_answer[0][2] =  J; Sz_answer[0][3] =  0; 
    Sz_answer[1][0] =  0; Sz_answer[1][1] = -h; Sz_answer[1][2] =  0; Sz_answer[1][3] =  J;
    Sz_answer[2][0] =  J; Sz_answer[2][1] =  0; Sz_answer[2][2] =  h; Sz_answer[2][3] =  0;
    Sz_answer[3][0] =  0; Sz_answer[3][1] =  J; Sz_answer[3][2] =  0; Sz_answer[3][3] = -h;

    np.testing.assert_array_equal(hamiltonian(t_test_second), Sz_answer)

def test_onsite_potential():
    h = 0.5
    J = 1.0
    N = 2
    hamiltonian = Hamiltonian(N=N,tagged=-1)
    hamiltonian.add_H_1body(Sx, coef=h)
    hamiltonian.add_H_2body(Sz, Sz, coef=J)
    np.testing.assert_array_equal(hamiltonian.onsite_potential(1), h*np.kron(Sx,np.eye(2,dtype=np.complex128)))

    J = 1.0
    N = 2
    hamiltonian = Hamiltonian(N=N,tagged=-1)
    hamiltonian.add_H_1body(Sx, coef=np.cos, time_depend=True)
    hamiltonian.add_H_2body(Sz, Sz, coef=J)
    t_test = 2.0 * np.pi
    h = np.cos(t_test)
    np.testing.assert_array_equal(hamiltonian.onsite_potential(1,t_test), h*np.kron(Sx,np.eye(2,dtype=np.complex128)))

