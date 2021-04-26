# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:12:23 2021

@author: anika
"""

# %% matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import MyTicToc as mt
import pandas as pd
#from scipy import integrate

# %% Import Data WieringermeerData_Meteo.xlsx (Date, Precip, Evap, Temp)

MeteoData = pd.read_csv (r'C:\Users\anika\Documents\Environmental Engineering\Modelling Coupled Processes\Assignment 1\WieringermeerData_Meteo.csv')
# print (MeteoData)

LeachateData = pd.read_csv (r'C:\Users\anika\Documents\Environmental Engineering\Modelling Coupled Processes\Assignment 1\WieringermeerData_LeachateProduction.csv')
# print (LeachateData)

# data with same dates selected

Qdr = LeachateData.iloc[:, 1]  # Leachate output [m^3/day] 
Jrfday = MeteoData.iloc[-(len(Qdr) + 1) : -1, 1]  # precipitation [m/day]
pEv = MeteoData.iloc[-(len(Qdr) + 1) : -1, 2]  # Evaporation [m/day]

""" Dates now line up """

# %% Parameters TODO

a = 500         # hydraulic ocnductivity, ex: 1.1 * Qdrain_max THE CURRENT NUMBER IS A PLACEHOLDER!!!
bcl = 1.25      # Some value > 1
bwb = 1.25
Sclmin = 1      # placeholder
Sclmax = 100    # placeholder
Swbmin = 1      # placeholder
Swbmax = 100    # placeholder
Ccrop = 0.9     # Cropfactor (researched. ref: Transport Phenomena)
Sevmin = 2      # placeholder
Sevmax = 10     # placeholder
Beta0 = 0.9     # placeholder
tstep = 24      # placeholder

# %% Definition of Rate Equation
# S[0] = Scl, S[1] = Swd

def dSdt(t, S):
    """ Return the rate of change of the storages """
    Scl = S[0]
    Swb = S[1]
    Seffcl = (Scl-Sclmin)/(Sclmax - Sclmin)
    Seffwb = (Swb-Swbmin)/(Swbmax - Swbmin)
    Lcl = a * Seffcl**bcl
    Lwb = a * Seffwb**bwb
    
    if Scl < Sevmin:        # TODO: Check this format for an if-statement
        fred = 0
    elif Scl > Sevmax:
            fred = 1
    else: fred = (Scl - Sevmin)/(Sevmax - Sevmin)
    
    E = pEv * Ccrop * fred    # TODO: find how to get times, find 
    
    Beta = Beta0 * Seffcl
    
    Jrf  = Jrfday / tstep   # Can we include t in this? Or do we define that somewhere else?
    # See brightspace for more on what to do with t
    
    dScldt = Jrf - Lcl - E    # function that extracts value from time (not an integer). Make sure to divide rainfall over day. Outflow data form hourly => daily
    dSwdt = (1 - Beta) * Lcl - Lwb
    Qdrain = Beta * Lcl + Lwb
    
    return np.array([dScldt,
                     dSwdt,
                     Qdrain])
    
# %%

def main():
    # Definition of output times
    #tOut = list(range(1, len(Qdr) + 1))        # time [days]
    tOut = np.arange(0, len(Qdr), 1/tstep)      # time as (a fraction of) days
#    nOut = np.shape(tOut)[0]                   # Do we need this?

    # Initial case, 
    S0 = np.array([1, 1, 1])   # Inital case. What should these be? TODO
    mt.tic()
    t_span = [tOut[0], tOut[-1]]
    SODE = sp.integrate.solve_ivp(dSdt, t_span, S0, t_eval=tOut, 
                                  method='RK45', vectorized=True, 
                                  rtol=1e-5 )   # Will calculate states from ODE solver
    # infodict['message']                     # >>> 'Integration successful.'
    
    # CAll on function again to get Qdrain based on the solved data. Compare with measurements
    
    SclODE = SODE.y[0,:]
    SwdODE = SODE.y[1,:]
    QdrainODE = SODE.y[2,:]
    
# %%     
    '''EULER - cut out for now'''

    '''EULER Predictor Corrector - cut out for now '''

    '''RungeKutta - Cut out for now'''
    
    
    
    # Instead of these figures, we should probably make a figure comparing the outputs for Qdrain to the measured Qdrain
    # We might even be able to create a function that minimizes the difference between selected Qdrain outputs and measured Qdrains

    # Plot results with matplotlib    
    plt.figure()
    plt.plot(tOut, SclODE, 'r-', label='LclODE')
    plt.plot(tOut, SwdODE, 'b-', label='LwdODE')
    plt.plot(tOut, QdrainODE, 'b-', label='BODE')
#    plt.plot(tOut, rEuler, 'g+', label='REuler')
#    plt.plot(tOut, fEuler, 'm+', label='FEuler')
#    plt.plot(tOut, rPC, 'rx', label='RPC')
#    plt.plot(tOut, fPC, 'bx', label='FPC')
#    plt.plot(tOut, rRK, 'g.', label='RRK')
#    plt.plot(tOut, fRK, 'm.', label='FRK')

    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.title('Evolution of fox and rabbit populations')
    # f1.savefig('rabbits_and_foxes_1.png')
    plt.show()

    plt.figure()
    plt.plot(fODE, rODE, 'b-', label='ODE')
#    plt.plot(fEuler, rEuler, 'b+', label='Euler')
#    plt.plot(fPC, rPC, 'r-', label='Predictor Corrector')
#    plt.plot(fRK, rRK, 'g-', label='Runge Kutta')

    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Foxes')    
    plt.ylabel('Rabbits')
    plt.title('Evolution of fox and rabbit populations')
    # f2.savefig('rabbits_and_foxes_2.png')
    plt.show()


if __name__ == "__main__":
    main()

