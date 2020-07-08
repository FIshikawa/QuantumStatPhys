import sys
import os
include_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'include')
sys.path.append(include_file)
import datetime
import collections
import numpy as np
from tools import LogManager
from open_system_experiments import experiment 
from physical_operators import *
from hamiltonian import Hamiltonian
from density_matrix import DensityMatrix

if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)

    experimental_condi = []
    experimental_condi.append('Observe Relaxation : ED : XXZ-Heisenberg model')
    experimental_condi.append('Hamiltonian : '\
                                    '- J ( Sx Sx + Sy Sy + V Sz Sz) - h Sz')

    param_dict = collections.OrderedDict()
    param_dict['result_dir']='test/'
    param_dict['result_thermalize'] = \
                            param_dict['result_dir'] + 'result_thermalize.dat' 
    param_dict['result_timedev'] = \
                            param_dict['result_dir'] + 'result_timedev.dat' 
    param_dict['N']=3
    param_dict['J']=1.0
    param_dict['h']=0.5
    param_dict['V']=1.0
    param_dict['J_int']=1.0
    param_dict['h_external']=0.5
    param_dict['Sx_init']=1.0
    param_dict['dt']=1.0E-3
    param_dict['t']=1*param_dict['dt']
    param_dict['N_time']=1
    param_dict['N_measure']=1
    param_dict['T']= 1 / param_dict['dt']#1.0
    param_dict['N_thermalize']= int( 1 / param_dict['T'] * param_dict['dt'])
    param_dict['tagged'] = (param_dict['N']-1)//2
    param_dict['engine'] = False
    param_dict['relaxation'] = False
    param_dict['test_mode'] = False
    param_dict['discrete'] = False
    param_dict['integrator'] = 'EulerMethod'

    counter = 1
    if(argc > counter): 
        param_dict['result_dir'] = str(argv[counter])
    counter += 1
    if(argc > counter): 
        param_dict['N'] = int(argv[counter])
    counter += 1
    if(argc > counter): 
        param_dict['J'] = float(argv[counter])
    counter += 1
    if(argc > counter):
        param_dict['h'] = float(argv[counter])
    counter += 1
    if(argc > counter):
        param_dict['V'] = float(argv[counter])
    counter += 1
    if(argc > counter): 
        param_dict['J_int'] = float(argv[counter])
    counter += 1
    if(argc > counter):
        param_dict['h_external'] = float(argv[counter])
    counter += 1
    if(argc > counter):
        param_dict['Sx_init'] = float(argv[counter])
    counter += 1
    if(argc > counter):
        param_dict['t'] = float(argv[counter])
    counter += 1
    if(argc > counter):
        param_dict['N_time'] = int(argv[counter])
    counter += 1
    if(argc > counter):
        param_dict['N_measure'] = int(argv[counter])
    counter += 1
    if(argc > counter):
        param_dict['T'] = float(argv[counter])
    counter += 1
    if(argc > counter):
        param_dict['tagged'] = int(argv[counter])
    counter += 1
    if(argc > counter):
        param_dict['engine'] = True if('True' in str(argv[counter])) else False
    counter += 1
    if(argc > counter):
        param_dict['relaxation'] = \
                True if('True' in str(argv[counter])) else False
    counter += 1
    if(argc > counter):
        param_dict['discrete'] = \
                True if('True' in str(argv[counter])) else False
    counter += 1
    if(argc > counter):
        param_dict['test_mode'] = \
                True if('True' in str(argv[counter])) else False
    counter += 1
    if(argc > counter):
        param_dict['integrator'] = str(argv[counter]) 

    integrators_name_list = ['EulerMethod','RungeKutta4th',
                             'RungeKutta3rd','RungeKutta2nd']
    if(not param_dict['integrator'] in integrators_name_list):
        raise KeyError('{} is not expected '\
                '(EulerMethod, RungeKutta2nd, RungeKutta3rd, RungeKutta4th'
                .format(param_dict['integrator']))

    param_dict['dt'] = param_dict['t'] / float(param_dict['N_time'])
    param_dict['result_thermalize'] = \
                       param_dict['result_dir'] + 'result_thermalize.dat' 
    param_dict['result_timedev'] = \
                       param_dict['result_dir'] + 'result_timedev.dat' 
    param_dict['result_band'] = param_dict['result_dir'] + 'result_band.dat' 
    param_dict['N_thermalize']= int( 1 / param_dict['T'] * param_dict['dt'])

    log_file = param_dict['result_dir'] + "condi.dat"

    if(os.path.exists(log_file)):
        os.remove(log_file) 
    loger = LogManager(log_file,stdout=True)

    start_time = "Experiment start : " \
                 + datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') + "\n"
    loger.log_write(start_time)

    loger.log_write('[Experiment Conditions]')
    for key in experimental_condi:
        loger.log_write('  {}'.format(key))
    loger.log_write('[parameters]')
    for key in param_dict.keys():
        loger.log_write('  {0} : {1}'.format(key, param_dict[key]))

    # each parameters set 
    tagged = int(param_dict['tagged'])
    N = param_dict['N']
    J = param_dict['J']
    h = param_dict['h']
    V = param_dict['V']
    h_external = param_dict['h_external'] 
    J_int = param_dict['J_int'] 
    Sx_init = param_dict['Sx_init'] 
    relaxation = param_dict['relaxation']
    engine = param_dict['engine']
    discrete = param_dict['discrete']

    # set external force 
    external_force = lambda t : -h_external*np.cos(np.pi*t)

    # set hamiltonian
    if(relaxation):
        hamiltonian_thermalize = Hamiltonian(N=N, tagged=tagged)
    else:
        hamiltonian_thermalize = Hamiltonian(N=N, tagged=tagged)
        hamiltonian_thermalize.add_H_1body(Sz, 
                                           coef=external_force(0),
                                           position=tagged)
        hamiltonian_thermalize.add_H_2body(Sx, 
                                           Sx, 
                                           coef=-J_int,
                                           position=[tagged-1,tagged])
        hamiltonian_thermalize.add_H_2body(Sy, 
                                           Sy, 
                                           coef=-J_int,
                                           position=[tagged-1,tagged])
        hamiltonian_thermalize.add_H_2body(Sz, 
                                           Sz, 
                                           coef=-J_int*V,
                                           position=[tagged-1,tagged])
    hamiltonian_thermalize.add_H_1body(Sz, coef=-h)
    hamiltonian_thermalize.add_H_2body(Sx, Sx, coef=-J)
    hamiltonian_thermalize.add_H_2body(Sy, Sy, coef=-J)
    hamiltonian_thermalize.add_H_2body(Sz, Sz, coef=-J*V)

    hamiltonian_total = Hamiltonian(N=N, tagged=tagged)
    hamiltonian_total.add_H_1body(Sz, coef=-h)
    hamiltonian_total.add_H_2body(Sx, Sx, coef=-J)
    hamiltonian_total.add_H_2body(Sy, Sy, coef=-J)
    hamiltonian_total.add_H_2body(Sz, Sz, coef=-J*V)
    hamiltonian_total.add_H_2body(Sx, 
                                  Sx, 
                                  coef=-J_int,
                                  position=[tagged-1,tagged])
    hamiltonian_total.add_H_2body(Sy, 
                                  Sy, 
                                  coef=-J_int,
                                  position=[tagged-1,tagged])
    hamiltonian_total.add_H_2body(Sz, 
                                  Sz, 
                                  coef=-J_int*V,
                                  position=[tagged-1,tagged])
    if(engine):
        if(not discrete):
            raise ValueError('must use discrete time develop in engine mode')
        hamiltonian_total.add_H_1body(Sz, 
                                      coef=external_force,
                                      time_depend=True,
                                      position=tagged)
    else:
        hamiltonian_total.add_H_1body(Sz, 
                                      coef=external_force(0),
                                      position=tagged)

    # set density matrix
    rho = DensityMatrix(N=N,
                        tagged=tagged,
                        relaxation=relaxation,
                        Sx_init=Sx_init)

    experiment(
                param_dict,
                loger,
                rho,
                hamiltonian_thermalize,
                hamiltonian_total
              )

    end_time = "Experiment finish : " + \
                datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') + "\n" 
    loger.log_write(end_time)
