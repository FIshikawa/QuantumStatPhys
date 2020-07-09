import copy
import numpy as np
from physical_operators import *

class DensityMatrix:
    def __init__(self, N, tagged=-1,relaxation=False,Sx_init=1.0):
        self.density_matrix = np.zeros((np.power(2,N),np.power(2,N)),
                                        dtype=np.complex128)
        self.tagged = tagged
        self.N = N
        if(Sx_init > 1):
            raise ValueError('Sx_init must be lower than 1')
        self.Sx_init = np.float64(Sx_init)
        self.initialize(relaxation=relaxation)

    def initialize(self,relaxation=False):
        if(relaxation):
            density_matrix = np.identity(np.power(2,self.N),
                                         dtype=np.complex128)
        else:
            density_right = np.eye(np.power(2,self.tagged),
                                          dtype=np.complex128)
            density_left = np.eye(np.power(2,self.N-self.tagged-1),
                                         dtype=np.complex128)
            density_tagged = 0.5 * np.eye(2,dtype=np.complex128) \
                                    + 0.5 * self.Sx_init * Sx

            density_matrix = np.kron(density_tagged,density_right)
            density_matrix = np.kron(density_left,density_matrix)
        self.density_matrix = density_matrix / np.trace(density_matrix)
  
    def calculate_expectation_of_1body(self, O=Sz, index=-1):
        N = self.N
        operator_total= np.zeros((np.power(2,N), np.power(2,N)), 
                                  dtype=np.complex128)
        if(index == -1):
            for i in range(N):
                operator_right = np.eye(np.power(2,i),dtype=np.complex128)
                operator_left = np.eye(np.power(2,N-i-1),dtype=np.complex128)
                operator_temp = \
                        np.kron(operator_left,np.kron(O,operator_right))
                operator_total += operator_temp
        else:
            operator_right = np.eye(np.power(2,index),dtype=np.complex128)
            operator_left = np.eye(np.power(2,N-index-1),dtype=np.complex128)
            operator_temp = np.kron(operator_left,np.kron(O,operator_right))
            operator_total += operator_temp 

        return np.trace(np.dot(operator_total,self.density_matrix)).real 

    def calculate_expectation_of_2body(self, O1=Sz, O2=Sz, dist=1, index=-1):
        N = self.N
        operator_total= np.zeros((np.power(2,N), np.power(2,N)), 
                                  dtype=np.complex128)
        if(index == -1):
            for i in range(N-dist-1):
                operator_right = np.eye(np.power(2,i),dtype=np.complex128)
                operator_middle = np.eye(np.power(2,dist),dtype=np.complex128)
                operator_left = np.eye(np.power(2,N-i-2-dist),
                                       dtype=np.complex128)
                operator_temp = np.kron(O1,operator_right)
                operator_temp = np.kron(operator_middle,operator_temp)
                operator_temp = np.kron(O2,operator_temp)
                operator_temp = np.kron(operator_left,operator_temp)
                operator_total += operator_temp
        elif(index > -1 and index + dist -1 < N):
            operator_right = np.eye(np.power(2,index),dtype=np.complex128)
            operator_middle = np.eye(np.power(2,dist-1),dtype=np.complex128)
            operator_left = np.eye(np.power(2,N-index-1-dist),
                                   dtype=np.complex128)
            operator_temp = np.kron(O1,operator_right)
            operator_temp = np.kron(operator_middle,operator_temp)
            operator_temp = np.kron(O2,operator_temp)
            operator_temp = np.kron(operator_left,operator_temp)
            operator_total += operator_temp
        else:
            raise ValueError('index {} and dist {} is over whole size'
                              .format(index, dist))

        return np.trace(np.dot(operator_total,self.density_matrix)).real 

    def trace_of_density_matrix(self):
        return np.trace(self.density_matrix)

    def expectation_calc(self,whole_operator):
        return np.trace(np.dot(whole_operator,self.density_matrix)).real 

    def copy(self):
        return copy.deepcopy(self)

    def reduced_density_matrix(self, left, right):
        N = self.N
        if(left < 0 or N-1 < right):
            raise ValueError('area {} to {} is over defined system [0 , N-1]'
                              .format(left,right))
        density_matrix = self.density_matrix.copy()
        self.reduce_left = left 
        self.reduce_right = right
        area_size = right - left + 1 
        reduced_density = \
                       np.zeros((np.power(2,area_size),np.power(2,area_size)),
                                 dtype=np.complex128)
        matrix_basis = np.zeros(np.power(4,area_size),dtype=np.complex128)
        left_reduction = np.eye(np.power(2,left),dtype=np.complex128)
        right_reduction = np.eye(np.power(2,N-right-1),dtype=np.complex128)
        for i in range(len(matrix_basis)):
            matrix_basis[i] = 1
            matrix_basis = np.reshape(matrix_basis,
                                 (np.power(2,area_size),np.power(2,area_size)))
            total_reduce_basis = np.kron(left_reduction,
                                         np.kron(matrix_basis,right_reduction))
            reduced_density += \
                    np.trace(np.dot(density_matrix,total_reduce_basis)) \
                        * matrix_basis
            matrix_basis = np.reshape(matrix_basis,np.power(4,area_size))
            matrix_basis[i] = 0
        return reduced_density

    def entanglement_entropy(self, left, right):
        reduced_density = self.reduced_density_matrix(left, right)
        D = np.linalg.eigvalsh(reduced_density)
        entanglement_entropy = 0
        for eigenvalue in D:
            if(eigenvalue > 0):
                entanglement_entropy -= eigenvalue * np.log(eigenvalue)
                if(np.isnan(entanglement_entropy)):
                    print('{} make nan'.format(eigenvalue))
                    entanglement_entropy = 0
            else:
                print('{} is not positive eigenvalu'.format(eigenvalue))
        return entanglement_entropy



