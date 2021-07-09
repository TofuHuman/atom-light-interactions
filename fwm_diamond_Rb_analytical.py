# Rubidium Atom-Ensemble Wavelength Conversion Scheme
# Pulses are considered by F.T of density matrix solutions

# Phys.Rev.A 82, 023815 (2010)
# Efficiency of light-frequency conversion in an atomic ensemble
# H. H. Jen and T. A. B. Kennedy

# States |0>, |1>, |2>, |3> in counter-clockwise direction
# forming the letter 'N'. |0> <----> |3> is input/idler transition
# |2> <-----> |1> is signal/output transition.
# All energy interactions are semi-classically treated.
# Input/Output transitions are considered to be very weak. (Quantized Coupling) 

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from scipy import integrate
from mpl_toolkits.mplot3d import axes3d

# Setting coupling equation parameters at each frequency, given control 
# field strengths and power :

# plotting parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 4
plt.rcParams.update({'figure.autolayout': True})

def set_parameters(Delta_omega_i, Omega_a, Omega_b, Delta_1, Delta_b):

    Delta_omega_s = Delta_omega_i - Delta_1 + Delta_b    #omega_s - omega_12
    Delta_2 =       Delta_1 + Delta_omega_s              #omega_a + omega_s - omega_2
   
    # T parameters in steady-state equations
    T_01 = gamma_01/2 - 1j*Delta_1
    T_02 = gamma_2/2 - 1j*Delta_2
    T_03 = gamma_03/2 - 1j*Delta_omega_i
    T_12 = (gamma_01 + gamma_2)/2 - 1j*Delta_omega_s
    T_13 = (gamma_01 + gamma_03)/2 + 1j*Delta_1 - 1j*Delta_omega_i

    # steady state values
    sigma_11_s = abs(Omega_a)**2/(Delta_1**2 + gamma_01**2/4 + 2*abs(Omega_a)**2)
    sigma_00_s = 1 - sigma_11_s
    sigma_01_s = 1j*Omega_a/(gamma_01/2 - 1j*Delta_1)*(1 - 2*sigma_11_s)

    # D parameter
    D = T_12*T_03 + T_12*((abs(Omega_a)**2)/T_13 + abs(Omega_b)**2/T_02) + T_03*((abs(Omega_a))**2/T_02 + (abs(Omega_b))**2/T_13) + ((abs(Omega_a))**2 - (abs(Omega_b))**2)**2/(T_02*T_13)

    # constants for coupled propagation equations
    alpha_i = (-N*(abs(g_i))**2)/(c*D)*(sigma_00_s*(T_12 + (abs(Omega_a))**2/T_02 + (abs(Omega_b))**2/T_13) - (1j*Omega_a*sigma_01_s.conjugate())/T_13*(T_12 + ((abs(Omega_a))**2 - (abs(Omega_b))**2)/T_02))
    beta_s = (-N*(abs(g_s))**2)/(c*D)*(sigma_11_s*(T_03 + (abs(Omega_a))**2/T_13 + (abs(Omega_b))**2/T_02) - (1j*Omega_a.conjugate()*sigma_01_s)/T_02*(T_03 + ((abs(Omega_a))**2 - (abs(Omega_b))**2)/T_13))
    kappa_s = (-N*g_i*g_s.conjugate())/(c*D)*(sigma_00_s*(Omega_a.conjugate()*Omega_b/T_02 + Omega_a.conjugate()*Omega_b/T_13) + (1j*Omega_b*sigma_01_s.conjugate())/(T_13)*(T_03 + ((abs(Omega_b))**2 - (abs(Omega_a))**2)/T_02))
    kappa_i = (-N*g_s*g_i.conjugate())/(c*D)*(sigma_11_s*(Omega_b.conjugate()*Omega_a/T_02 + Omega_b.conjugate()*Omega_a/T_13) + (1j*Omega_b.conjugate()*sigma_01_s)/(T_02)*(T_12 + ((abs(Omega_b))**2 - (abs(Omega_a))**2)/T_13))

    q = (-alpha_i + beta_s)/2
    w = np.sqrt(q**2 + (kappa_i*kappa_s))

    return alpha_i, beta_s, kappa_s, kappa_i, q, w

# constants
hbar = 1.054571e-34
c = 2.99792458e8                    # speed of light
qe = -1.60217646e-19                # charge of electron
a0 = 5.291772e-11                   # bohr radius
eps0 = 8.854e-12                    # vacuum permittivity

# frequencies and angular frequencies of applied and actual transitions
lambda_a = 780.241e-9               # 0 <----> 1    # 5S1/2 F = 1 to 5P3/2 F = 2
lambda_s = 1367e-9                  # 1 <----> 2    # 5P3/2 F = 2 to 6S1/2 F = 1
lambda_b = 1324e-9                  # 2 <----> 3    # 5P1/2 F = 1 to 6S1/2 F = 1
lambda_i = 794.978e-9               # 3 <----> 0    # 5S1/2 F = 1 to 5P1/2 F = 2

omega_a = (2*np.pi*c)/(lambda_a) + 4.271e9 - 72.911e6
omega_s = (2*np.pi*c)/(lambda_s) + 72.911e6 - 0.8e9
omega_i = (2*np.pi*c)/(lambda_i) + 4.271e9 + 302.246e6
omega_b = (2*np.pi*c)/(lambda_b) - 302.246e6 - 0.8e9
  
#print('Net Energy Difference:', (omega_a + omega_s) - (omega_b + omega_i))

# dipole matrix elements   (check)
d_01 = 4.227*qe*a0
d_03 = 2.992*qe*a0
d_12 = 4.257*qe*a0  #(check)
d_23 = 2.918*qe*a0  #(check)

# data:
# steck: https://steck.us/alkalidata/rubidium87numbers.pdf
# safranova: https://arxiv.org/pdf/physics/0307057.pdf (85/87?)
# herold: https://arxiv.org/pdf/1208.4291.pdf

# coupling parameters of input and signal photons
sigma = 3*lambda_i**2/(4*np.pi)
opd = 150
L = 6e-3
rho = opd/(sigma*L)

V = 1           # cancels
N = rho*V

g_s = d_12/hbar*np.sqrt(hbar*omega_s/(2*eps0*V))
g_i = d_03/hbar*np.sqrt(hbar*omega_i/(2*eps0*V))

# decay rate, decay transition values (rad/s units)
gamma_01 = 1/(26.24e-9)            # given
gamma_02 = 0
gamma_03 = 1/(27.7e-9)             # given
gamma_12 = gamma_03/2.76           # given
gamma_13 = 0
gamma_32 = gamma_03/5.38           # given
gamma_2 = gamma_12 + gamma_32

# default
Omega_a = 33*gamma_03
Omega_b = 20*gamma_03
Delta_1 = 39*gamma_03
Delta_b = 2*gamma_03

# returning coupled equation parameters 
############################################################

def all_return(inputs, opd):
    delta_i, Omega_a, Omega_b, Delta_1, Delta_b = inputs
    alpha_i, beta_s, kappa_s, kappa_i, q, w = set_parameters(delta_i, Omega_a, Omega_b, Delta_1, Delta_b)
    return alpha_i, beta_s, kappa_s, kappa_i, q, w

def eta_conversion(inputs, opd):
    delta_i, Omega_a, Omega_b, Delta_1, Delta_b = inputs
    alpha_i, beta_s, kappa_s, kappa_i, q, w = set_parameters(delta_i, Omega_a, Omega_b, Delta_1, Delta_b)
    eta_conversion = (abs((kappa_s/(2*w))*np.exp((alpha_i + beta_s)*L/2)*(np.exp(w*L) - np.exp(-w*L))))**2
    return eta_conversion

def transmission(inputs, opd):
    delta_i, Omega_a, Omega_b, Delta_1, Delta_b = inputs
    alpha_i, beta_s, kappa_s, kappa_i, q, w = set_parameters(delta_i, Omega_a, Omega_b, Delta_1, Delta_b)
    transmission = (abs(np.exp((alpha_i + beta_s)*L/2)/(2*w*(w + q))*(kappa_s*kappa_i*np.exp(w*L) + (q + w)**2*np.exp(-w*L))))**2
    return transmission

###########################################################

# gaussian pulse 
def gauss(x, s):
    return np.exp(-((x)**2. / (2. * s**2.)))

# square gaussian pulse
def square_pulse_gaussian(E_it, pulse_time, Tn):
    
    T_index = np.linspace(0, Tn, Tn)
    n = Tn

    T0 = pulse_time*1e9/200  
    T0 = T0/2
    T10 = 0.5 - T0 
    T11 = 0.5 + T0

    T1 = n*(T10 - 0.15)
    T2 = n*T10 + 1
    T3 = n*T10 + 1
    T4 = n*T11 + 1
    T5 = n*T11
    T6 = n*(T11 + 0.25)

    DT = n*0.03

    E_it[:, 0] = np.where((np.floor(T_index) > T1) & (np.floor(T_index) < T2), E_it[:,0] + gauss(T3 - T_index, DT), E_it[:,0])
    E_it[:, 0] = np.where((np.floor(T_index) > T3) & (np.floor(T_index) < T4), E_it[:,0] + 1, E_it[:,0])
    E_it[:, 0] = np.where((np.floor(T_index) > T5) & (np.floor(T_index) < T6), E_it[:,0] + gauss(T5 - T_index, DT), E_it[:,0])
    
    return E_it

# square sine pulse
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

        E_i[:, 0] = np.where((np.floor(T_index) > T1) & (np.floor(T_index) < T2), E_i[:, 0] + (0.5*(1 + np.sin(np.pi*(T_index - T1a)*10/Nt))), E_i[:, 0])
        E_i[:, 0] = np.where((np.floor(T_index) > T3) & (np.floor(T_index) < T4), E_i[:, 0] + 1, E_i[:, 0])
        E_i[:, 0] = np.where((np.floor(T_index) > T5) & (np.floor(T_index) < T6), E_i[:, 0] + (0.5*(1 + np.sin(np.pi*(T6a - T_index)*10/Nt))), E_i[:, 0])

    if (pulse_time == 15e-9):
        # rising and falling for 10ns 
        
        T1 = int(0.095/0.55*Nt)      # n*(T10 - DT)
        T2 = int(0.205/0.55*Nt) + 1 # n*T10 + 1
        T3 = int(0.205/0.55*Nt)   # n*T10 
        T4 = int(0.345/0.55*Nt) + 1 # n*T11 + 1
        T5 = int(0.345/0.55*Nt)     # n*T11
        T6 = int(0.455/0.55*Nt)      # n*(T11 + DT)

        T1a = int(0.15/0.55*Nt)
        T6a = int(0.40/0.55*Nt)

        E_i[:, 0] = np.where((np.floor(T_index) > T1) & (np.floor(T_index) < T2), E_i[:, 0] + (0.5*(1 + np.sin(np.pi*(T_index - T1a)*5/Nt))), E_i[:, 0])
        E_i[:, 0] = np.where((np.floor(T_index) > T3) & (np.floor(T_index) < T4), E_i[:, 0] + 1, E_i[:, 0])
        E_i[:, 0] = np.where((np.floor(T_index) > T5) & (np.floor(T_index) < T6), E_i[:, 0] + (0.5*(1 + np.sin(np.pi*(T6a - T_index)*5/Nt))), E_i[:, 0])

    return E_i
    
# pulse propagation 
def pulse_coupled_equation():

    opd = 150

    n = 10000
    Zn = n
    Fn = n
    Tn = n

    #pulse_time = 1e-9  
    pulse_time = 15e-9
    
    Z = np.linspace(0, L, Zn)
    
    # input and signal fields 
    E_if = np.zeros((Fn, Zn), dtype=complex)
    E_sf = np.zeros((Fn, Zn), dtype=complex)

    E_it = np.zeros((Tn, Zn), dtype=complex)
    E_st = np.zeros((Tn, Zn), dtype=complex)

    ## gaussian pulse input
    # net_time = 5*pulse_time
    # T_ns = np.linspace(-1*net_time/2, net_time/2, Tn)   
    # E_it[:, 0] = gauss(T_ns, pulse_time/2)

    ## square wave sine input
    if (pulse_time == 100e-9):
        net_time = 200*1e-9

    if (pulse_time == 15e-9):
        net_time = 55*1e-9
    E_it = square_pulse_sine_idler(E_it, n, pulse_time)
    
    T = np.linspace(0, net_time*1e9, Tn)   
    plt.plot(T, E_it[:, 0])
    plt.xlabel('ns')
    plt.ylabel('Norm.')
    plt.show()

    from scipy.fft import fft, ifft, fftshift, fftfreq

    E_if[:,0] = fft(E_it[:,0])
    F = fftfreq(n, net_time/n)
    # plt.plot(F, E_if[:, 0])
    # plt.show()

    delta_i = -21*gamma_03
    delta_i_values = delta_i - 2*np.pi*F 

    for i in range(Fn):
        inputs = delta_i_values[i], Omega_a, Omega_b, Delta_1, Delta_b
        alpha_i, beta_s, kappa_s, kappa_i, q, w = all_return(inputs, opd)

        beta_s = beta_s - 1j*delta_i_values[i]/c 
        alpha_i = alpha_i - 1j*delta_i_values[i]/c

        z = Zn - 1
        l = Z[z]

        eta_conv = ((kappa_s/(2*w))*np.exp((alpha_i + beta_s)*l/2)*(np.exp(w*l) - np.exp(-w*l)))
        trans = (np.exp((alpha_i + beta_s)*l/2)/(2*w*(w + q))*(kappa_s*kappa_i*np.exp(w*l) + (q + w)**2*np.exp(-w*l)))

        E_sf[i, z] =  E_if[i, 0]*eta_conv
        E_if[i, z] =  E_if[i, 0]*trans 
        
    E_it[:,0] = ifft(E_if[:,0])
    E_it[:,Zn - 1] = ifft(E_if[:,Zn - 1])
    E_st[:,Zn - 1] = ifft(E_sf[:,Zn - 1])

    plt.plot(T, abs(E_it[:,Zn-1])**2, 'b--', label='Transmitted Input, z = L')
    plt.plot(T, abs(E_st[:,Zn-1])**2, 'r--', label='Transmitted Signal, z = L')
    plt.plot(T, abs(E_it[:,Zn-1])**2 + abs(E_st[:,Zn-1])**2, 'm--', label='Net Transmitted, z = L')
    plt.plot(T, abs(E_it[:,0])**2, 'b', label='Input, z = 0')
    
    plt.xlabel("ns")
    plt.ylabel(r"$|E|^2$  [Norm.]")
    plt.legend(prop={"size":10.5}, loc = "upper right")
    plt.show()

    signal_integrated = np.trapz(abs(E_st[:,Zn-1])**2, dx=net_time/n)
    input_integrated = np.trapz(abs(E_it[:,0])**2, dx=net_time/n)
    print('Pulse Conversion Efficiency:', signal_integrated/input_integrated)
    
def scan_delta_i():

    number = 1000
    delta_omega_i_list = np.linspace(-80*gamma_03,80*gamma_03, number)
    delta_omega_i_index = np.linspace(-80*gamma_03/(1e6*2*np.pi), 80*gamma_03/(1e6*2*np.pi), number)
    opd = 150

    eta_conversion_scan_delta_i = np.zeros(number)
    transmission_input_scan_delta_i = np.zeros(number)

    for i in range(number):
        delta_i = delta_omega_i_list[i]
        inputs = delta_i, Omega_a, Omega_b, Delta_1, Delta_b
        eta_conversion_scan_delta_i[i] = abs(eta_conversion(inputs, opd))
        transmission_input_scan_delta_i[i] = abs(transmission(inputs, opd))

    plt.plot(delta_omega_i_index, eta_conversion_scan_delta_i, 'b', label = 'Conversion')
    plt.plot(delta_omega_i_index, transmission_input_scan_delta_i, 'r', label = 'Transmission')
    plt.legend()
    plt.xlabel(r"$\Delta_i$ [MHz]")
    plt.ylabel(r"$\eta_{C, T}$")
    plt.show()

#scan_delta_i()

pulse_coupled_equation()



