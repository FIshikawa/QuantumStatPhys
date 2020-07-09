import sys
import os
import argparse
include_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'include')
sys.path.append(include_file)
import datetime
import numpy as np
from tools import LogManager
from open_system_experiments import experiment 
from physical_operators import *
from hamiltonian import Hamiltonian
from density_matrix import DensityMatrix

if __name__ == '__main__':
    experimental_condi = []
    experimental_condi.append('Observe Relaxation : ED : XXZ-Heisenberg model')
    experimental_condi.append('Hamiltonian : '\
                                    '- J ( Sx Sx + Sy Sy + V Sz Sz) - g Sz')
    parser = argparse.ArgumentParser(description=experimental_condi[0])

    parser.add_argument('--result_directory',required=True,action='store',
                        type=str,help='Result directory',default='./test/')
    parser.add_argument('--N',required=True,action='store',type=int,
                        help='System size',default=3)
    parser.add_argument('--J',required=True,action='store',type=float,
                        help='Interaction strength',default=1.0)
    parser.add_argument('--g',required=True,action='store',type=float,
                        help='Static external field strength',default=0.5)
    parser.add_argument('--V',required=True,action='store',type=float,
                        help='Anisotropy field strength',default=1.0)
    parser.add_argument('--J_int',required=True,action='store',type=float,
                        help='System-bath coupling',default=1.0)
    parser.add_argument('--g_external',required=True,action='store',type=float,
                        help='Time depend external field strength',default=0.5)
    parser.add_argument('--Sx_init',required=True,action='store',type=float,
                        help='Initial state of system',default=1.0)
    parser.add_argument('--t',required=True,action='store',type=float,
                        help='time for development',default=1.0)
    parser.add_argument('--N_time',required=True,action='store',type=int,
                        help='Number of time step',default=10)
    parser.add_argument('--T',required=True,action='store',type=float,
                        help='Temperture of Bath',default=10)
    parser.add_argument('--N_measure',required=True,action='store',type=int,
                        help='Number of measure step',default=1)
    parser.add_argument('--tagged',required=True,action='store',type=int,
                        help='Site of tagged spin',default=1)
    parser.add_argument('--engine',required=True,action='store',type=str,
                        help='Flag of heat engine calc',default='False')
    parser.add_argument('--relaxation',required=True,
                        action='store',type=str,
                        help='Flag of whole system relaxation',default='False')
    parser.add_argument('--discrete',required=True,
                        action='store',type=str,
                        help='Flag of discrete time development',
                        default='False')
    parser.add_argument('--test_mode',required=True,
                        action='store',type=str,
                        help='Flag of test mode',default='False')
    parser.add_argument('--integrator',required=True,action='store',type=str,
                        help='Integrator for time development',
                        default='EulerMethod')

    param_dict = vars(parser.parse_args())

    integrators_name_list = ['EulerMethod','RungeKutta4th',
                             'RungeKutta3rd','RungeKutta2nd']
    if(not param_dict['integrator'] in integrators_name_list):
        raise KeyError('{} is not expected '\
                '(EulerMethod, RungeKutta2nd, RungeKutta3rd, RungeKutta4th'
                .format(param_dict['integrator']))

    param_dict['dt'] = param_dict['t'] / float(param_dict['N_time'])
    param_dict['result_thermalize'] = \
          os.path.join(param_dict['result_directory'],'result_thermalize.dat') 
    param_dict['result_timedev'] = \
          os.path.join(param_dict['result_directory'],'result_timedev.dat')
    param_dict['result_band'] = \
          os.path.join(param_dict['result_directory'],'result_band.dat') 
    param_dict['N_thermalize']= int( 1 / param_dict['T'] / param_dict['dt'])

    for flag in ['engine', 'relaxation', 'discrete', 'test_mode']:
        param_dict[flag] = \
                True if param_dict[flag] in ['True','true'] else False

    log_file = os.path.join(param_dict['result_directory'], 'condi.dat')

    if(os.path.exists(log_file)):
        os.remove(log_file) 
    loger = LogManager(log_file,stdout=True)

    start_time = 'Experiment start : ' \
                 + datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S') 
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
    g = param_dict['g']
    V = param_dict['V']
    g_external = param_dict['g_external'] 
    J_int = param_dict['J_int'] 
    Sx_init = param_dict['Sx_init'] 
    relaxation = param_dict['relaxation']
    engine = param_dict['engine']
    discrete = param_dict['discrete']

    # set external force 
    external_force = lambda t : -g_external*np.cos(np.pi*t)

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
    hamiltonian_thermalize.add_H_1body(Sz, coef=-g)
    hamiltonian_thermalize.add_H_2body(Sx, Sx, coef=-J)
    hamiltonian_thermalize.add_H_2body(Sy, Sy, coef=-J)
    hamiltonian_thermalize.add_H_2body(Sz, Sz, coef=-J*V)

    hamiltonian_total = Hamiltonian(N=N, tagged=tagged)
    hamiltonian_total.add_H_1body(Sz, coef=-g)
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
