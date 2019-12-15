import numpy as np
import copy
class EachHamiltonian:
    def __init__(self,operator,body,position):
        if(isinstance(position,int)):
            self.position =[position]
        elif(isinstance(position,list)):
            self.position = copy.deepcopy(position)
        else:
            raise ValueError('input position is not expected (int or list)')

        if(isinstance(operator,np.ndarray)):
            self.operator = operator.copy()
        else:
            self.operator = operator
        self.body = body 

    def __call__(self,t=None):
        if(t is None):
            return self.operator.copy()
        else:
            return self.operator(t)
