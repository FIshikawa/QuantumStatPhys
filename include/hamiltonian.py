import numpy as np
from each_hamiltonian import EachHamiltonian
class Hamiltonian:
    def __init__(self, N, tagged=-1):
        self.N = N
        self.tagged = tagged

        self.H_list = []
        self.H_time_depend_list = []

        self.H_kind = 0
        self.H_time_depend_kind = 0

        self.H_time_independ = None
        self.time_depend = False

        self.onsite_hamiltonian = None
        self.onsite_position = -1

    def _add_H(self,H_temp,body,coef,time_depend,position):
        if(position < 0):
            if(self.tagged < 0):
                position = [x for x in range(self.N-body+1)]
            else:
                limited_list = [self.tagged - x for x in range(body)]
                position = [x for x in range(self.N-body+1) if not  x in limited_list ]
        if(time_depend):
            H_each = EachHamiltonian(operator=lambda t : coef(t)*H_temp,
                                    body=body,
                                    position=position)
            self.H_time_depend_list.append(H_each)
            self.H_time_depend_kind += 1
            self.time_depend = True
        else:
            H_each = EachHamiltonian(operator=coef*H_temp,
                                    body=body,
                                    position=position)
            self.H_list.append(H_each)
            self.H_kind += 1

    def add_H_1body(self, H_1body, coef=1.0, time_depend=False, position=-1):
        H_temp = np.zeros((2, 2), dtype=np.complex128)
        H_temp = H_1body.copy()
        body = 1
        self._add_H(H_temp,body,coef,time_depend,position)

    def add_H_2body(self, H_2body1, H_2body2, coef=1.0, time_depend=False, position=-1):
        H_temp = np.zeros((4, 4), dtype=np.complex128)
        H_temp = np.kron(H_2body1, H_2body2)
        body = 2
        self._add_H(H_temp,body,coef,time_depend,position)

    def add_H_3body(self, H_3body1, H_3body2, H_3body3, coef=1.0, time_depend=False, position=-1):
        H_temp = np.zeros((8, 8), dtype=np.complex128)
        H_temp = np.kron(H_3body1, np.kron(H_3body2, H_3body3))
        body = 3
        self._add_H(H_temp,body,coef,time_depend,position)

    def operator_reshape(self,position,body,H_each):
        N = self.N
        H_right = np.eye(np.power(2,position),dtype=np.complex128)
        H_left  = np.eye(np.power(2,N-position-body),dtype=np.complex128)
        H_each_total = np.kron(H_left, np.kron(H_each,H_right))
        if(H_each_total.shape[0] != np.power(2,N) or H_each_total.shape[1] != np.power(2,N)):
            raise ValueError('H_each inconsist with input body parameter')
        return H_each_total

    def onsite_potential(self,position,t=None):
        N = self.N
        if(self.onsite_position != position):
            onsite_hamiltonian = np.zeros((np.power(2,N),np.power(2,N)),dtype=np.complex128)
            for H_each in self.H_list:
                if(H_each.body == 1):
                    onsite_hamiltonian += self.operator_reshape(position,1,H_each())
            self.onsite_hamiltonian = onsite_hamiltonian.copy()
            self.onsite_position = position
        else:
            onsite_hamiltonian = self.onsite_hamiltonian.copy()
        if(t is not None and self.time_depend):
            for H_each in self.H_time_depend_list:
                if(H_each.body == 1):
                    onsite_hamiltonian += self.operator_reshape(position,1,H_each(t))
        return onsite_hamiltonian

    def set_hamiltonian(self,t=None):
        N = self.N
        H_total = np.zeros((np.power(2,N), np.power(2,N)), dtype=np.complex128)
        if(t is None):
            H_kind = self.H_kind
            H_list = self.H_list
        else:
            H_kind = self.H_time_depend_kind
            H_list = self.H_time_depend_list

        for H_each in H_list:
            body = H_each.body 
            interaction = np.zeros((np.power(2,N), np.power(2,N)), dtype=np.complex128)
            if(t is None):
                H_temp = H_each()
            else:
                H_temp = H_each(t)
            for position in H_each.position:
                interaction_temp = self.operator_reshape(position,body,H_temp)
                interaction += interaction_temp
            H_total += interaction

        return H_total

    def __call__(self,t=0):
        if(self.H_time_independ is None):
            self.H_time_independ = self.set_hamiltonian()
        if(self.time_depend):
            hamiltonian = self.H_time_independ.copy()
            hamiltonian += self.set_hamiltonian(t=t)
            return hamiltonian
        else:
            return self.H_time_independ.copy()
