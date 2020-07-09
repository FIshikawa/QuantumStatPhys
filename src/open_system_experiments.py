import sys
import os
include_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'include')
sys.path.append(include_file)
import numpy as np
from physical_operators import *
from exact_diagonalization import ExactDiagonalization

def physical_value_calc(t,file_name, density_matrix, hamiltonian_series):
    N = density_matrix.N
    tagged   = density_matrix.tagged

    Sz_ave = density_matrix.calculate_expectation_of_1body(O=Sz)/np.float64(N)
    Sx_ave = density_matrix.calculate_expectation_of_1body(O=Sx)/np.float64(N)
    Sy_ave = density_matrix.calculate_expectation_of_1body(O=Sy)/np.float64(N)
    SzSz_ave = density_matrix.calculate_expectation_of_2body(
                                                          O1=Sz, 
                                                          O2=Sz, 
                                                          dist=1
                                                          )/np.float64(N)
    SxSx_ave = density_matrix.calculate_expectation_of_2body(
                                                          O1=Sx, 
                                                          O2=Sx, 
                                                          dist=1
                                                          )/np.float64(N)
    SySy_ave = density_matrix.calculate_expectation_of_2body(
                                                          O1=Sy, 
                                                          O2=Sy, 
                                                          dist=1
                                                          )/np.float64(N)

    Sz_tagged   = density_matrix.calculate_expectation_of_1body(O=Sz,
                                                                index=tagged)
    Sx_tagged   = density_matrix.calculate_expectation_of_1body(O=Sx,
                                                                index=tagged)
    Sy_tagged   = density_matrix.calculate_expectation_of_1body(O=Sy,
                                                                index=tagged)
    SzSz_tagged = density_matrix.calculate_expectation_of_2body(O1=Sz, 
                                                                O2=Sz, 
                                                                dist=1, 
                                                                index=tagged)
    SxSx_tagged = density_matrix.calculate_expectation_of_2body(O1=Sx, 
                                                                O2=Sx, 
                                                                dist=1, 
                                                                index=tagged)
    SySy_tagged = density_matrix.calculate_expectation_of_2body(O1=Sy, 
                                                                O2=Sy, 
                                                                dist=1, 
                                                                index=tagged)

    total_energy = density_matrix.expectation_calc(hamiltonian_series[0])
    bath_energy  = density_matrix.expectation_calc(hamiltonian_series[1])
    system_energy  = density_matrix.expectation_calc(hamiltonian_series[2](t))
    interaction_energy = total_energy - (bath_energy + system_energy)

    entanglement_entropy = \
            density_matrix.entanglement_entropy(left=tagged,right=tagged)

    result_list = [t, Sz_tagged, Sz_ave, Sx_tagged, Sx_ave, Sy_tagged, Sy_ave, 
                   SzSz_tagged, SzSz_ave,\
                   SxSx_tagged, SxSx_ave, SySy_tagged, SySy_ave, 
                   total_energy, bath_energy, system_energy,\
                   interaction_energy, entanglement_entropy]

    f = open(file_name, "a")
    for value in result_list:
        f.write("%.16e " % value)
    f.write("\n")
    f.close()


def experiment(param_dict,loger,rho,hamiltonian_thermalize,hamiltonian_total):
    print('[Start Simulation]')
    N = param_dict['N']
    dt = np.float64(param_dict['dt'])
    t_end = param_dict['t']
    beta = 1 / param_dict['T']
    discrete = param_dict['discrete']
    engine = param_dict['engine']
    tagged = rho.tagged
    N_measure = param_dict['N_measure']
    test_mode = param_dict['test_mode']
    integrator = param_dict['integrator']

    if(integrator == 'EulerMethod'):
        from euler_method import EulerMethod as Integrator
    elif(integrator == 'RungeKutta2nd'):
        from runge_kutta_2nd import RungeKutta2nd as Integrator
    elif(integrator == 'RungeKutta3rd'):
        from runge_kutta_3rd import RungeKutta3rd as Integrator
    elif(integrator == 'RungeKutta4th'):
        from runge_kutta_4th import RungeKutta4th as Integrator
    else:
        raise KyeError('{} does not exists in expected integrators'
                        .format(integrator))

    print('integrator : {}'.format(integrator))

    # set hamiltonians 
    hamiltonian_whole = hamiltonian_total()
    hamiltonian_bath = hamiltonian_thermalize()
    if(engine):
        hamiltonian_system = \
                lambda t : hamiltonian_total.onsite_potential(tagged,t)
    else:
        hamiltonian_system = \
                lambda t : hamiltonian_total.onsite_potential(tagged)
    hamiltonian_series = \
            [hamiltonian_whole, hamiltonian_bath, hamiltonian_system]

    # set thermalization  
    thermalization = \
            ExactDiagonalization(N=N, hamiltonian=hamiltonian_thermalize())

    # set time develop
    if(discrete):
        if(engine):
            time_develop = Integrator(N=N,hamiltonian=hamiltonian_total)
        else:
            time_develop = \
                    Integrator(N=N,hamiltonian=(lambda t: hamiltonian_whole))
    else:
        time_develop = ExactDiagonalization(N=N, hamiltonian=hamiltonian_whole)

    result_thermalize = param_dict['result_thermalize']
    result_timedev = param_dict['result_timedev']
    result_band = param_dict['result_band']

    result_form = 'Sz_tagged Sz_ave Sx_tagged Sx_ave Sy_tagged Sy_ave '\
                  'SzSz_tagged SzSz_ave SxSx_tagged SxSx_ave ' \
                  'SySy_tagged SySy_ave '\
                  'total_energy bath_energy system_energy interaction_energy'\
                  ' entanglement_entropy\n'

    for file_name in [result_thermalize, result_timedev,result_band]:
        f = open(file_name, 'w')
        if(file_name == result_thermalize):
            f.write('#output_form beta ' + result_form)
        elif(file_name == result_timedev):
            f.write('#output_form t ' + result_form)
        else:
            band_calc = \
                    ExactDiagonalization(N=N, hamiltonian=hamiltonian_whole)
            f.write('#output_form k Energy\n')
            for i in range(len(band_calc.D)):
                f.write('{} {:.16e} \n'.format(i, band_calc.D[i]))
        f.close()

    #thermalize process
    print('thermalize process')
    t = 0.0
    print('check Energy_init')
    print('beta : {0}, {1}'.format(t,rho.expectation_calc(hamiltonian_whole)))
    file_name = result_thermalize
    physical_value_calc(t,file_name, rho, hamiltonian_series)
    rho_init = rho.copy()
    if(test_mode):
        d_beta = dt
        N_beta = int(beta / dt)
    else:
        d_beta = np.float64(beta / 5.0)
        N_beta = 5
    for i in range(N_beta):
        t += d_beta
        rho = thermalization(rho_init,-0.5j*t)
        if(i % N_measure == 0 and test_mode):
            physical_value_calc(t,file_name, rho, hamiltonian_series)
        else:
            physical_value_calc(t,file_name, rho, hamiltonian_series)
    print('check Energy_fin')
    print('beta : {0}, {1}'.format(t,rho.expectation_calc(hamiltonian_whole)))

    #time develop process
    print('time-develop process')
    t = 0.0
    print('check Energy_init')
    print('t : {0}, {1}'.format(t,rho.expectation_calc(hamiltonian_whole)))
    file_name = result_timedev
    physical_value_calc(t,file_name, rho, hamiltonian_series)
    if(not discrete):
        rho_thermalized = rho.copy() 
    for i in range(int(t_end / dt)):
        t += dt
        if(discrete):
            rho = time_develop(dt,rho,t)
        else:
            rho = time_develop(rho_thermalized,t)
        if(i % N_measure == 0):
            physical_value_calc(t,file_name, rho, hamiltonian_series)
    print('check Energy_fin')
    print('t : {0}, {1}'.format(t,rho.expectation_calc(hamiltonian_whole)))

    print('[Finish Simulation]')
