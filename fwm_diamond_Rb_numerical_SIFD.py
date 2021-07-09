# Pulse propagation for Rb (attempt to replicate results
# shared in: (Phys. Rev. A 82, 023815)
# Following the implicit central difference method
# (Comp. Phys. Comm, 29 (1983) 211—225)
# 15 ns and 100 ns input pulses.

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# plotting parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 4
plt.rcParams.update({'figure.autolayout': True})

# propagation parameters 
pulse_time = 15e-9 #15e-9 #100e-9
#pulse_type = 'gaussian'
pulse_type = 'square'
time_varying_control = 1

# basic constants (SI units) 
hbar = 1.054571e-34
c = 2.99792458e8    
eps0 = 8.854e-12                      # vacuum permittivity
a0 =  5.291772e-11                    # bohr radius
qe = 1.60217646e-19                  # charge of electron
 
# angular frequencies of applied and actual transitions

lambda_a = 780e-9               # 0 <----> 1
lambda_s = 1367e-9              # 1 <----> 2
lambda_b = 1324e-9              # 2 <----> 3
lambda_i = 795e-9               # 3 <----> 0

omega_a = (2*np.pi*c)/(lambda_a)
omega_s = (2*np.pi*c)/(lambda_s)
omega_i = (2*np.pi*c)/(lambda_i)
omega_b = (2*np.pi*c)/(lambda_b)

Ec_i = np.sqrt(rho*hbar*omega_i/(2*eps0))
Ec_s = np.sqrt(rho*hbar*omega_s/(2*eps0))
#print(Ec_i)

# z = z/Lc
# t = t/Tc
# Omega = Tc*Omega
# E_s,i = E/Ec

# atomic and applied field properties
# frequency originally in (rad/sec) to scaled time units 

gamma_03 = (1/(27.7*1e-9))*Tc
gamma_01 = (1/(26.2*1e-9))*Tc

gamma_12 = gamma_03/2.76  
gamma_32 = gamma_03/5.38
gamma_2 = gamma_12 + gamma_32
print(gamma_03)

Omega_b = 20*gamma_03

delta_1 = 39*gamma_03
delta_b = 2*gamma_03
delta_i = -21*gamma_03
delta_s = delta_b + delta_i - delta_1
delta_2 = delta_1 + delta_s

g_factor = 1.035**2

# scaling of time, fields and space (SI units)

Tc = np.sqrt(gamm)
Lc = 26e-3
Tc = 0.086*1e-9                         
rho = 1.7e17                          # (N_atoms/V) 
d_03 = 4.230*qe*a0

# initializations
# E (fields) are defined on a 2D (z, t) grid. (starting from a known E_i(0, t), the 
# system evolves). rho (density matrix) are kept track of 3 element array (represnting old, current and new) 
# using 11 variables that are updated in a for loop. 

# check time in co-moving frame

if (pulse_time == 100e-9):
    net_time = 200*1e-9
    T = net_time/Tc  # (dimensionless)

if (pulse_time == 15e-9):
    if (pulse_type == 'gaussian'):
        net_time = 5*pulse_time
    else: 
        net_time = 55*1e-9
    T = net_time/Tc

net_length = 6e-3
L = net_length/Lc

dz = 0.001
dt = 0.5               #/(2*np.pi)  # assuming 

Nz = int(L/dz)
Nt = int(T/dt)

print('Space steps', Nz)
print('Time steps', Nt)

E_i = np.zeros((Nz, Nt), dtype=complex)
E_s = np.zeros((Nz, Nt), dtype=complex)

# initial conditions

# E_i[0, :] = input waveform 
# E_s[0, :] = 0 

# gaussian pulse 
def gauss(x, s):
    return np.exp(-((x)**2. / (2. * s**2.)))

def square_pulse_sine_idler(E_i, Nt, pulse_time):

    T_index = np.linspace(0, Nt, Nt)
    n = Nt
    
    if (pulse_time == 100e-9):
        # rising and falling for 20ns  
        
        T1 = int(0.15*Nt)      # n*(T10 - DT)
        T2 = int(0.25*Nt) + 1 # n*T10 + 1
        T3 = int(0.25*Nt)     # n*T10 
        T4 = int(0.75*Nt) + 1 # n*T11 + 1
        T5 = int(0.75*Nt)     # n*T11
        T6 = int(0.85*Nt)      # n*(T11 + DT)

        T1a = int(0.2*Nt)
        T6a = int(0.8*Nt)

        E_i[0, :] = np.where((np.floor(T_index) > T1) & (np.floor(T_index) < T2), E_i[0, :] + (0.5*(1 + np.sin(1*np.pi*(T_index - T1a)/(0.2/2*Nt)))), E_i[0, :])
        E_i[0, :] = np.where((np.floor(T_index) > T3) & (np.floor(T_index) < T4), E_i[0, :] + 1, E_i[0, :])
        E_i[0, :] = np.where((np.floor(T_index) > T5) & (np.floor(T_index) < T6), E_i[0, :] + (0.5*(1 + np.sin(np.pi*(T6a - T_index)/(0.2/2*Nt)))), E_i[0, :])

    # if (pulse_time == 15e-9):
    #     # rising and falling for 10ns 
        
    #     T1 = int(0.1/0.55*Nt)      # n*(T10 - DT)
    #     T2 = int(0.20/0.55*Nt) + 1 # n*T10 + 1
    #     T3 = int(0.20/0.55*Nt)   # n*T10 
    #     T4 = int(0.35/0.55*Nt) + 1 # n*T11 + 1
    #     T5 = int(0.35/0.55*Nt)     # n*T11
    #     T6 = int(0.45/0.55*Nt)      # n*(T11 + DT)

    #     T1a = int(0.15/0.55*Nt)
    #     T6a = int(0.40/0.55*Nt)

    #     E_i[0, :] = np.where((np.floor(T_index) > T1) & (np.floor(T_index) < T2), E_i[0, :] + (0.5*(1 + np.sin(np.pi*(T_index - T1a)/(0.1/0.55*Nt)))), E_i[0, :])
    #     E_i[0, :] = np.where((np.floor(T_index) > T3) & (np.floor(T_index) < T4), E_i[0, :] + 1, E_i[0, :])
    #     E_i[0, :] = np.where((np.floor(T_index) > T5) & (np.floor(T_index) < T6), E_i[0, :] + (0.5*(1 + np.sin(np.pi*(T6a - T_index)/(0.1/0.55*Nt)))), E_i[0, :])

    if (pulse_time == 15e-9):
    
        # rising and falling for 7.5ns 

        T1 = int(0.125/0.55*Nt)      # n*(T10 - DT)
        T2 = int(0.20/0.55*Nt) + 1 # n*T10 + 1
        T3 = int(0.20/0.55*Nt)   # n*T10 
        T4 = int(0.35/0.55*Nt) + 1 # n*T11 + 1
        T5 = int(0.35/0.55*Nt)     # n*T11
        T6 = int(0.425/0.55*Nt)      # n*(T11 + DT)

        T1a = int(0.1625/0.55*Nt)
        T6a = int(0.3875/0.55*Nt)

        E_i[0, :] = np.where((np.floor(T_index) > T1) & (np.floor(T_index) < T2), E_i[0, :] + (0.5*(1 + np.sin(np.pi*(T_index - T1a)/(0.075/0.55*Nt)))), E_i[0, :])
        E_i[0, :] = np.where((np.floor(T_index) > T3) & (np.floor(T_index) < T4), E_i[0, :] + 1, E_i[0, :])
        E_i[0, :] = np.where((np.floor(T_index) > T5) & (np.floor(T_index) < T6), E_i[0, :] + (0.5*(1 + np.sin(np.pi*(T6a - T_index)/(0.075/0.55*Nt)))), E_i[0, :])

    return E_i

def square_pulse_sine_control(Omega_a, Nt):

    T_index = np.arange(Nt)
    n = Nt

    if (pulse_time == 100e-9):
        T_actual = np.linspace(0, 200e-9, Nt)
        # rising and falling for 10ns 
        
        T1 = int(0.25/2*Nt)         # n*(T10 - DT)
        T2 = int(0.35/2*Nt) + 1     # n*T10 + 1
        T3 = int(0.35/2*Nt)         # n*T10 
        T4 = int(0.165/0.2*Nt) + 1  # n*T11 + 1
        T5 = int(0.165/0.2*Nt)      # n*T11
        T6 = int(0.175/0.2*Nt)      # n*(T11 + DT)

        T1a = int(0.3/2*Nt)
        T6a = int(0.17/0.2*Nt)

        Omega_a[:] = np.where((np.floor(T_index) > T1) & (np.floor(T_index) < T2), Omega_a[:] + (0.5*(1 + np.sin(np.pi*(T_index - T1a)/(0.1/2*Nt)))), Omega_a[:])
        Omega_a[:] = np.where((np.floor(T_index) > T3) & (np.floor(T_index) < T4), Omega_a[:] + 1, Omega_a[:])
        Omega_a[:] = np.where((np.floor(T_index) > T5) & (np.floor(T_index) < T6), Omega_a[:] + (0.5*(1 + np.sin(np.pi*(T6a - T_index)/(0.1/2*Nt)))), Omega_a[:])

    if (pulse_time == 15e-9):
        T_actual = np.linspace(0, 55e-9, Nt)
        # rising and falling for 10ns 
        
        T1 = int(0.075/0.55*Nt)         # n*(T10 - DT)
        T2 = int(0.125/0.55*Nt)  + 1    # n*T10 + 1
        T3 = int(0.125/0.55*Nt)         # n*T10 
        T4 = int(0.425/0.55*Nt)  + 1    # n*T11 + 1
        T5 = int(0.425/0.55*Nt)         # n*T11
        T6 = int(0.475/0.55*Nt)         # n*(T11 + DT)

        T1a = int(0.1/0.55*Nt)
        T6a = int(0.45/0.55*Nt)

        Omega_a[:] = np.where((np.floor(T_index) > T1) & (np.floor(T_index) < T2), Omega_a[:] + (0.5*(1 + np.sin(np.pi*(T_index - T1a)/(0.05/0.55*Nt)))), Omega_a[:])
        Omega_a[:] = np.where((np.floor(T_index) > T3) & (np.floor(T_index) < T4), Omega_a[:] + 1, Omega_a[:])
        Omega_a[:] = np.where((np.floor(T_index) > T5) & (np.floor(T_index) < T6), Omega_a[:] + (0.5*(1 + np.sin(np.pi*(T6a - T_index)/(0.05/0.55*Nt)))), Omega_a[:])
    
    return Omega_a

# check; a small enough E_i input 
norm_E_i =  (1/(Ec_i*Tc))*(hbar/d_03)*(0.1*gamma_03)

plt.figure()
T_plotter = np.linspace(0, net_time*1e9, Nt)

if (time_varying_control == 1):
    Omega_a = np.zeros(Nt)
    Omega_a = (33*gamma_03)*square_pulse_sine_control(Omega_a, Nt)
    plt.plot(T_plotter, Omega_a[:]/np.max(Omega_a), 'k', label='$\Omega_a$ Laser')
else: 
    Omega_a = 33*gamma_03*np.ones(Nt)

if (pulse_type == 'gaussian'):
    ## gaussian pulse input
    net_time = 5*pulse_time
    T_ns = np.linspace(-1*net_time/2, net_time/2, Nt)   
    E_i[0, :] = norm_E_i*gauss(T_ns, pulse_time/2)

else: 
    # square pulse input
    E_i = norm_E_i*square_pulse_sine_idler(E_i, Nt, pulse_time)

plt.plot(T_plotter, E_i[0, :]/np.max(E_i), 'b', label='Input')
plt.xlabel('ns')
plt.ylabel('Norm.')
plt.legend()
plt.show()

real_rho_01 = np.zeros(3)
imag_rho_01 = np.zeros(3)
real_rho_12 = np.zeros(3)
imag_rho_12 = np.zeros(3)
real_rho_02 = np.zeros(3)
imag_rho_02 = np.zeros(3)
real_rho_00 = np.zeros(3)
real_rho_11 = np.zeros(3)
real_rho_22 = np.zeros(3)
real_rho_33 = np.zeros(3)
real_rho_13 = np.zeros(3)
imag_rho_13 = np.zeros(3)
real_rho_03 = np.zeros(3)
imag_rho_03 = np.zeros(3)
real_rho_32 = np.zeros(3)
imag_rho_32 = np.zeros(3)

# initial conditions
real_rho_32_initial_guess = 0 
imag_rho_32_initial_guess = 0 

real_rho_12_initial_guess = 0 
imag_rho_12_initial_guess = 0 

real_rho_02_initial_guess = 0 
imag_rho_02_initial_guess = 0 

real_rho_11_initial_guess = 0 #Omega_a_value**2/(delta_1**2 + gamma_01**2/4 + 2*Omega_a_value**2) 
real_rho_22_initial_guess = 0 
real_rho_33_initial_guess = 0 
real_rho_00_initial_guess = 1 - real_rho_11_initial_guess

real_rho_01_initial_guess = 0 #np.real(1j*Omega_a_value/(gamma_01/2 - 1j*delta_1)*(1 - 2*real_rho_11_initial_guess))
imag_rho_01_initial_guess = 0 #np.imag(1j*Omega_a_value/(gamma_01/2 - 1j*delta_1)*(1 - 2*real_rho_11_initial_guess))

real_rho_13_initial_guess = 0 
imag_rho_13_initial_guess = 0 

real_rho_03_initial_guess = 0 
imag_rho_03_initial_guess = 0 

real_rho_32_initial_guess = 0 
imag_rho_32_initial_guess = 0 

# simultaneous nonlinear equation solver:
# 11 equations in terms to solve for: new rho elements (9), E_i, E_s 

# assumption (?)  
# Omega^(*) = Omega

def maxwell_bloch_coupled_equations(inputs, *prv_inputs):

    real_rho_01, imag_rho_01, real_rho_12, imag_rho_12, real_rho_02, imag_rho_02, real_rho_00, real_rho_11, real_rho_22, real_rho_33, real_rho_13, imag_rho_13, real_rho_03, imag_rho_03, real_rho_32, imag_rho_32, real_E_i, imag_E_i, real_E_s, imag_E_s = inputs
    real_rho_01_old, imag_rho_01_old, real_rho_12_old, imag_rho_12_old, real_rho_02_old, imag_rho_02_old, real_rho_00_old, real_rho_11_old, real_rho_22_old, real_rho_33_old, real_rho_13_old, imag_rho_13_old, real_rho_03_old, imag_rho_03_old, real_rho_32_old, imag_rho_32_old, real_E_i_back, imag_E_i_back, real_E_s_back, imag_E_s_back, Omega_a = prv_inputs

    delta_s = delta_b + delta_i - delta_1
    delta_2 = delta_1 + delta_s

    return [real_rho_01 - real_rho_01_old - 2*dt*(-1*gamma_01/2*(real_rho_01 + real_rho_01_old)/2 - delta_1*(imag_rho_01 + imag_rho_01_old)/2 - (imag_rho_02 + imag_rho_02_old)/2*(real_E_s + real_E_s_back)/2 + (real_rho_02 + real_rho_02_old)/2*(imag_E_s + imag_E_s_back)/2 + (real_rho_13 + real_rho_13_old)/2*(imag_E_i + imag_E_i_back)/2 - (imag_rho_13 + imag_rho_13_old)/2*(real_E_i + real_E_i_back)/2),    
            imag_rho_01 - imag_rho_01_old - 2*dt*(-1*gamma_01/2*(imag_rho_01 + imag_rho_01_old)/2 + delta_1*(real_rho_01 + real_rho_01_old)/2 + Omega_a*(real_rho_00 + real_rho_00_old - real_rho_11 - real_rho_11_old)/2 + (real_rho_02 + real_rho_02_old)/2*(real_E_s + real_E_s_back)/2 + (imag_rho_02 + imag_rho_02_old)/2*(imag_E_s + imag_E_s_back)/2 - (real_rho_13 + real_rho_13_old)/2*(real_E_i + real_E_i_back)/2 - (imag_rho_13 + imag_rho_13_old)/2*(imag_E_i + imag_E_i_back)/2),
            real_rho_12 - real_rho_12_old - 2*dt*(-1*((gamma_01 + gamma_2)/2)*(real_rho_12 + real_rho_12_old)/2 - delta_s*(imag_rho_12 + imag_rho_12_old)/2 + Omega_a*(imag_rho_02 + imag_rho_02_old)/2 - (real_rho_11 + real_rho_11_old - real_rho_22 - real_rho_22_old)/2*(imag_E_s + imag_E_s_back)/2 - Omega_b*(imag_rho_13 + imag_rho_13_old)/2),
            imag_rho_12 - imag_rho_12_old - 2*dt*(-1*((gamma_01 + gamma_2)/2)*(imag_rho_12 + imag_rho_12_old)/2 + delta_s*(real_rho_12 + real_rho_12_old)/2 - 1*Omega_a*(real_rho_02 + real_rho_02_old)/2 + (real_rho_11 + real_rho_11_old - real_rho_22 - real_rho_22_old)/2*(real_E_s + real_E_s_back)/2 + Omega_b*(real_rho_13 + real_rho_13_old)/2),
            real_rho_02 - real_rho_02_old - 2*dt*(-1*(gamma_2)/2*(real_rho_02 + real_rho_02_old)/2  - delta_2*(imag_rho_02 + imag_rho_02_old)/2 + Omega_a*(imag_rho_12 + imag_rho_12_old)/2  - (imag_rho_01 + imag_rho_01_old)/2*(real_E_s + real_E_s_back)/2 - (real_rho_01 + real_rho_01_old)/2*(imag_E_s + imag_E_s_back)/2 - Omega_b*(imag_rho_03 + imag_rho_03_old)/2 + (real_rho_32 + real_rho_32_old)/2*(imag_E_i + imag_E_i_back)/2 + (imag_rho_32 + imag_rho_32_old)/2*(real_E_i + real_E_i_back)/2), 
            imag_rho_02 - imag_rho_02_old - 2*dt*(-1*(gamma_2)/2*(imag_rho_02 + imag_rho_02_old)/2 + delta_2*(real_rho_02 + real_rho_02_old)/2 - Omega_a*(real_rho_12 + real_rho_12_old)/2 + (real_rho_01 + real_rho_01_old)/2*(real_E_s + real_E_s_back)/2 - (imag_rho_01 + imag_rho_01_old)/2*(imag_E_s + imag_E_s_back)/2 + Omega_b*(real_rho_03 + real_rho_03_old)/2 - (real_rho_32 + real_rho_32_old)/2*(real_E_i + real_E_i_back)/2 + (imag_rho_32 + imag_rho_32_old)/2*(imag_E_i + imag_E_i_back)/2), 
            real_rho_11 - real_rho_11_old - 2*dt*(-1*gamma_01*(real_rho_11 + real_rho_11_old)/2 + gamma_12*(real_rho_22 + real_rho_22_old)/2 + 2*Omega_a*(imag_rho_01 + imag_rho_01_old)/2 - 2*((imag_rho_12 + imag_rho_12_old)/2*(real_E_s + real_E_s_back)/2 - (real_rho_12 + real_rho_12_old)/2*(imag_E_s + imag_E_s_back)/2)), 
            real_rho_22 - real_rho_22_old - 2*dt*(-1*gamma_2*(real_rho_22 + real_rho_22_old)/2 + 2*((imag_rho_12 + imag_rho_12_old)/2*(real_E_s + real_E_s_back)/2 - (real_rho_12 + real_rho_12_old)/2*(imag_E_s + imag_E_s_back)/2) + 2*Omega_b*(imag_rho_32 + imag_rho_32_old)/2),
            real_rho_33 - real_rho_33_old - 2*dt*(-1*gamma_03*(real_rho_33 + real_rho_33_old)/2 + gamma_32*(real_rho_22 + real_rho_22_old)/2 - 2*Omega_b*(imag_rho_32 + imag_rho_32_old)/2 + 2*((imag_rho_03 + imag_rho_03_old)/2*(real_E_i + real_E_i_back)/2 - (real_rho_03 + real_rho_03_old)/2*(imag_E_i + imag_E_i_back)/2)), 
            real_rho_13 - real_rho_13_old - 2*dt*(-1*((gamma_03 + gamma_01)/2)*(real_rho_13 + real_rho_13_old)/2 + (-1)*(delta_i - delta_1)*(imag_rho_13 + imag_rho_13_old)/2 + Omega_a*(imag_rho_03 + imag_rho_03_old)/2 + (imag_E_s + imag_E_s_back)/2*(real_rho_32 + real_rho_32_old)/2 - (real_E_s + real_E_s_back)/2*(imag_rho_32 + imag_rho_32_old)/2 - Omega_b*(imag_rho_12 + imag_rho_12_old)/2 - (imag_E_i + imag_E_i_back)/2*(real_rho_01 + real_rho_01_old)/2 + (real_E_i + real_E_i_back)/2*(imag_rho_01 + imag_rho_01_old)/2), 
            imag_rho_13 - imag_rho_13_old - 2*dt*(-1*((gamma_03 + gamma_01)/2)*(imag_rho_13 + imag_rho_13_old)/2 + (delta_i - delta_1)*(real_rho_13 + real_rho_13_old)/2 - Omega_a*(real_rho_03 + real_rho_03_old)/2 - (imag_E_s + imag_E_s_back)/2*(imag_rho_32 + imag_rho_32_old)/2 - (real_E_s + real_E_s_back)/2*(real_rho_32 + real_rho_32_old)/2 + Omega_b*(real_rho_12 + real_rho_12_old)/2 + (imag_E_i + imag_E_i_back)/2*(imag_rho_01 + imag_rho_01_old)/2 + (real_E_i + real_E_i_back)/2*(real_rho_01 + real_rho_01_old)/2),
            real_rho_03 - real_rho_03_old - 2*dt*(-1*gamma_03/2*(real_rho_03 + real_rho_03_old)/2 - delta_i*(imag_rho_03 + imag_rho_03_old)/2 + Omega_a*(imag_rho_13 + imag_rho_13_old)/2 - Omega_b*(imag_rho_02 + imag_rho_02_old)/2 - (real_rho_00 + real_rho_00_old - real_rho_33 - real_rho_33_old)/2*(imag_E_i + imag_E_i_back)/2),          
            imag_rho_03 - imag_rho_03_old - 2*dt*(-1*gamma_03/2*(imag_rho_03 + imag_rho_03_old)/2 + delta_i*(real_rho_03 + real_rho_03_old)/2 - Omega_a*(real_rho_13 + real_rho_13_old)/2 + Omega_b*(real_rho_02 + real_rho_02_old)/2 + (real_rho_00 + real_rho_00_old - real_rho_33 - real_rho_33_old)/2*(real_E_i + real_E_i_back)/2),          
            real_rho_32 - real_rho_32_old - 2*dt*(-1*((gamma_03 + gamma_2)/2)*(real_rho_32 + real_rho_32_old)/2 - delta_b*(imag_rho_32 + imag_rho_32_old)/2 - (imag_E_s + imag_E_s_back)/2*(real_rho_13 + real_rho_13_old)/2 + (real_E_s + real_E_s_back)/2*(imag_rho_13 + imag_rho_13_old)/2 + (imag_rho_02 + imag_rho_02_old)/2*(real_E_i + real_E_i_back)/2 - (real_rho_02 + real_rho_02_old)/2*(imag_E_i + imag_E_i_back)/2), 
            imag_rho_32 - imag_rho_32_old - 2*dt*(-1*((gamma_03 + gamma_2)/2)*(imag_rho_32 + imag_rho_32_old)/2 + delta_b*(real_rho_32 + real_rho_32_old)/2 + (real_E_s + real_E_s_back)/2*(real_rho_13 + real_rho_13_old)/2 + (imag_E_s + imag_E_s_back)/2*(imag_rho_13 + imag_rho_13_old)/2 - Omega_b*(real_rho_22 + real_rho_22_old - real_rho_33 - real_rho_33_old)/2 - (real_rho_02 + real_rho_02_old)/2*(real_E_i + real_E_i_back)/2 - (imag_rho_02 + imag_rho_02_old)/2*(imag_E_i + imag_E_i_back)/2),
            (real_rho_00 + real_rho_00_old)/2 + (real_rho_11 + real_rho_11_old)/2 + (real_rho_22 + real_rho_22_old)/2 + (real_rho_33 + real_rho_33_old)/2 - 1,
            real_E_i - real_E_i_back + dz*(imag_rho_03 + imag_rho_03_old),
            imag_E_i - imag_E_i_back - dz*(real_rho_03 + real_rho_03_old),
            (real_E_s - real_E_s_back) + g_factor*dz*(imag_rho_12 + imag_rho_12_old),
            (imag_E_s - imag_E_s_back) - g_factor*dz*(real_rho_12 + real_rho_12_old)]

index = Nz - 2

for i in range(index): #(Nz - 1): 
    i = i + 1
    print('Space Loop', i)

    for j in range(Nt - 1):
        # Time Loop
        j = j + 1

        if (i == 1):
          
            real_E_i_back = np.real(E_i[0, j])
            real_E_s_back = 0
            imag_E_i_back = 0
            imag_E_s_back = 0

        if (i > 1):
  
            real_E_i_back = np.real(E_i[i - 1, j])
            real_E_s_back = np.real(E_s[i - 1, j])
            imag_E_i_back = np.imag(E_i[i - 1, j])
            imag_E_s_back = np.imag(E_s[i - 1, j])


        if (j == 1):
            
            real_rho_32[0] = real_rho_32_initial_guess 
            imag_rho_32[0] = imag_rho_32_initial_guess
            real_rho_12[0] = real_rho_12_initial_guess
            imag_rho_12[0] = imag_rho_12_initial_guess
            real_rho_02[0] = real_rho_02_initial_guess
            imag_rho_02[0] = imag_rho_02_initial_guess 
            real_rho_00[0] = real_rho_00_initial_guess
            real_rho_11[0] = real_rho_11_initial_guess
            real_rho_22[0] = real_rho_22_initial_guess
            real_rho_33[0] = real_rho_33_initial_guess
            real_rho_01[0] = real_rho_01_initial_guess
            imag_rho_01[0] = imag_rho_01_initial_guess
            real_rho_13[0] = real_rho_13_initial_guess 
            imag_rho_13[0] = imag_rho_13_initial_guess
            real_rho_03[0] = real_rho_03_initial_guess
            imag_rho_03[0] = imag_rho_03_initial_guess 

        if (j > 1):

            # previous rho elements are calculated via mid-point average
            real_rho_12[0] = (real_rho_12[2] + real_rho_12[0])/2
            imag_rho_12[0] = (imag_rho_12[2] + imag_rho_12[0])/2
            real_rho_02[0] = (real_rho_02[2] + real_rho_02[0])/2
            imag_rho_02[0] = (imag_rho_02[2] + imag_rho_02[0])/2
            real_rho_00[0] = (real_rho_00[2] + real_rho_00[0])/2  
            real_rho_11[0] = (real_rho_11[2] + real_rho_11[0])/2 
            real_rho_22[0] = (real_rho_22[2] + real_rho_22[0])/2 
            real_rho_33[0] = (real_rho_33[2] + real_rho_33[0])/2 
            real_rho_01[0] = (real_rho_01[2] + real_rho_01[0])/2
            imag_rho_01[0] = (imag_rho_01[2] + imag_rho_01[0])/2
            real_rho_13[0] = (real_rho_13[2] + real_rho_13[0])/2
            imag_rho_13[0] = (imag_rho_13[2] + imag_rho_13[0])/2 
            real_rho_03[0] = (real_rho_03[2] + real_rho_03[0])/2
            imag_rho_03[0] = (imag_rho_03[2] + imag_rho_03[0])/2
            real_rho_32[0] = (real_rho_32[2] + real_rho_32[0])/2
            imag_rho_32[0] = (imag_rho_32[2] + imag_rho_32[0])/2 

        #initial = real_rho_01_initial_guess, imag_rho_01_initial_guess, real_rho_12_initial_guess, imag_rho_12_initial_guess, real_rho_02_initial_guess, imag_rho_02_initial_guess, real_rho_00_initial_guess, real_rho_11_initial_guess, real_rho_22_initial_guess, real_rho_33_initial_guess, real_rho_13_initial_guess, imag_rho_13_initial_guess, real_rho_03_initial_guess, imag_rho_03_initial_guess, real_rho_32_initial_guess, imag_rho_32_initial_guess, real_E_i_back, imag_E_i_back, real_E_s_back, imag_E_s_back
        #initial = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 
        
        prv_inputs = real_rho_01[0], imag_rho_01[0], real_rho_12[0], imag_rho_12[0], real_rho_02[0], imag_rho_02[0], real_rho_00[0], real_rho_11[0], real_rho_22[0], real_rho_33[0], real_rho_13[0], imag_rho_13[0], real_rho_03[0], imag_rho_03[0], real_rho_32[0], imag_rho_32[0], real_E_i_back, imag_E_i_back, real_E_s_back, imag_E_s_back, Omega_a[j]
        initial = real_rho_01[0], imag_rho_01[0], real_rho_12[0], imag_rho_12[0], real_rho_02[0], imag_rho_02[0], real_rho_00[0], real_rho_11[0], real_rho_22[0], real_rho_33[0], real_rho_13[0], imag_rho_13[0], real_rho_03[0], imag_rho_03[0], real_rho_32[0], imag_rho_32[0], real_E_i_back, imag_E_i_back, real_E_s_back, imag_E_s_back
        real_rho_01[2], imag_rho_01[2], real_rho_12[2], imag_rho_12[2], real_rho_02[2], imag_rho_02[2], real_rho_00[2], real_rho_11[2], real_rho_22[2], real_rho_33[2], real_rho_13[2], imag_rho_13[2], real_rho_03[2], imag_rho_03[2], real_rho_32[2], imag_rho_32[2], real_E_i_forward, imag_E_i_forward, real_E_s_forward, imag_E_s_forward = fsolve(maxwell_bloch_coupled_equations, initial, args=prv_inputs)
        
        # (temporary) forward field solutions
        E_i[i + 1, j] = real_E_i_forward + 1j*imag_E_i_forward
        E_s[i + 1, j] = real_E_s_forward + 1j*imag_E_s_forward
        #print('E_i_forward_solution', E_i[i + 1, j])

        # averaged current state field
        E_i[i, j] = 1/2*( E_i[i - 1, j] +  E_i[i + 1, j])
        E_s[i, j] = 1/2*( E_s[i - 1, j] +  E_s[i + 1, j])

        if (j == int(Nt/2)):
            print('E_i (current, averaged)', E_i[i, j])
            print('E_i (forward, solved)', E_i[i + 1, j])

# enter name of file to save E field data
np.save('E_i_full_test_100ns_new.npy', E_i)
np.save('E_s_full_test_100ns_new.npy', E_s)

norm_E_i_1 = np.max(np.abs(E_i))**2

plt.plot(T_plotter, np.abs(E_i[0, :])**2/norm_E_i, 'b', label='Input')
plt.plot(T_plotter, np.abs(E_s[index, :])**2/norm_E_i, 'r--', label='Transmitted Signal')
plt.plot(T_plotter, np.abs(E_i[index, :])**2/norm_E_i, 'b--', label='Transmitted Input')
#plt.plot(T_plotter, np.abs(E_i[Nz - 2, :])/np.max(np.abs(E_i[Nz - 2, :])), 'b--', label='Transmitted Input')
#plt.plot(T_plotter, np.abs(E_s[Nz - 2, :])/np.max(np.abs(E_s[Nz - 2, :])), 'r--', label='Transmitted Signal')
plt.legend()
plt.show()

plt.plot(T_plotter, np.abs(E_i[0, :])**2, 'b', label='Input')
plt.legend()
plt.show()
plt.plot(T_plotter, np.abs(E_i[index, :])**2, 'b--', label='Transmitted Input')
plt.legend()
plt.show()
plt.plot(T_plotter, np.abs(E_s[index, :])**2, 'r--', label='Transmitted Signal')
#plt.plot(T_plotter, np.abs(E_i[Nz - 2, :])/np.max(np.abs(E_i[Nz - 2, :])), 'b--', label='Transmitted Input')
#plt.plot(T_plotter, np.abs(E_s[Nz - 2, :])/np.max(np.abs(E_s[Nz - 2, :])), 'r--', label='Transmitted Signal')
plt.legend()
plt.show()