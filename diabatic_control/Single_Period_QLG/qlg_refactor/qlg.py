"""Core QLG (Quantum Lattice Gate) class.

Extracted and lightly cleaned from the notebook.
"""

from __future__ import annotations

import numpy as np
from numpy import exp
from math import factorial, pi, sqrt
from scipy.special import hyp1f1, genlaguerre, gamma
from qutip import num

class QLG():
    '''
    # the QLG class
    For a given Hamiltonian (a list of cnm on the Fock basis ,i.e, H = sum_{n,m} c_{c,m} |n><m|) 
    And constructing the time-dependent Hamiltonian accordingly in terms of Floquet engineering
    Notes: please see the anlytical expressions in the reference paper: https://www.nature.com/articles/s42005-025-02354-0
    '''
    def __init__(self, Np):
        '''
        :param Np: the photon number (cutoff dimension)
        '''
        self.Np = Np # the photon number cutoff
        self.E=1 # the energy gap 

        self.lamd = 1/4 # the effective hbar
        self.alpha =  2.34474 # #1.537851.53785#2.34474 # the sweet spot displacement for four-legged cat staes
        self.omega0=1 # the oscillator frequency (the unit of time, energy)
        self.Dtau= 2 * np.pi / self.omega0 ## the time period
        ki,kf,Nk = 10**(-5),30,64 # the k space discretization parameters
        self.klist = np.linspace(0, kf, Nk+1)+ ki
        self.dk = self.klist[1] - self.klist[0]
        self.H0  = self.lamd* self.omega0 * (num(Np) + 0.5) # the bare Hamiltonian

    def fnm(self,m,n,k,tau,lamd):  # real number
        if m-n>-1:
            fnm1 = np.exp(0.25*lamd*k**2)*((factorial(m)/factorial(n))**0.5)*(1j*np.exp(1j*tau)*((2/lamd)**0.5)/k)**(n-m)*(lamd/gamma(1+m-n))*hyp1f1(1+m,1+m-n,-0.5*k**2*lamd)
        else:
            fnm1 = np.exp(0.25*lamd*k**2)*((factorial(n)/factorial(m))**0.5)*(1j*np.exp(-1j*tau)* ((2/lamd)**0.5)/k)**(m-n)*(lamd/gamma(1+n-m))*hyp1f1(1+n,1+n-m,-0.5*k**2*lamd)
        return fnm1

    def fc(self,tau,k,cnm_list):
        fclist1 = np.sum([cmn*self.fnm(m,m1,k,tau,self.lamd) for cmn,m,m1 in cnm_list])
        return  fclist1
    
    def Vnmtc(self,n, m, tau,cnm_list):
        dk,klist,lamd = self.dk ,self.klist,self.lamd
        V_nmt = 0
        if m >= n:
            for k in  klist:
                V_nmt += dk * k * exp(-lamd * k ** 2 / 4) * sqrt(factorial(n ) / factorial(m )) * genlaguerre(
                    n , m - n)(0.5 * lamd * k ** 2) * np.real(
                    self.fc(tau,k,cnm_list)  * (sqrt(lamd * 0.5) * 1j * k) ** (m - n))
        else:
            for k in  klist:
                V_nmt += dk * k * np.exp(-lamd * k ** 2 / 4) * sqrt(factorial(m ) / factorial(n )) * genlaguerre(
                    m , n - m)(lamd * k ** 2 / 2) * np.real(
                    self.fc(tau,k,cnm_list)  * (sqrt(lamd * 0.5) * 1j * k) ** (n - m))
        return V_nmt

    def get_H(self,t,cnm_list,amp):
        return  amp*self.V_Cat_t(t,cnm_list)#H0.full()+

    def V_Cat_t(self,t,cnm_list):
        Np=self.Np
        V0 = np.zeros((Np, Np), dtype='complex128')
        for n in range(Np):
            for m in range(Np):
                V0[n,m] = self.Vnmtc(n, m, t,cnm_list)#+Vnmte(n, m,t,Delta)
        return V0

    ## ======= discrete the k space =========
 
    def Vnmtc_k(self,k,n, m, tau,cnm_list):
        lamd = self.lamd
        if m >= n:
            V_nmt = k * exp(-lamd * k ** 2 / 4) * sqrt(factorial(n ) / factorial(m )) * genlaguerre(
                n , m - n)(0.5 * lamd * k ** 2) * np.real(
                    self.fc(tau,k,cnm_list)  * (sqrt(lamd * 0.5) * 1j * k) ** (m - n))
        else:
            V_nmt = k * np.exp(-lamd * k ** 2 / 4) * sqrt(factorial(m ) / factorial(n )) * genlaguerre(
                m , n - m)(lamd * k ** 2 / 2) * np.real(
                    self.fc(tau,k,cnm_list)  * (sqrt(lamd * 0.5) * 1j * k) ** (n - m))
            # V_nmt_list.append(V_nmt)
        return  V_nmt

    def V_Cat_t_k(self,t,k,cnm_list):
        Np = self.Np
        V0 = np.zeros((Np, Np), dtype='complex128')
        for n in range(Np):
            for m in range(Np):
                V0[n,m] = self.Vnmtc_k(k,n, m, t,cnm_list)#+Vnmte(n, m,t,Delta)
        return V0

    def V_Cat_t_klist(self,t, klist,cnm_list):
        return np.array([self.V_Cat_t_k(t,k,cnm_list) for k in klist])
        
    def V_Cat_t1(self,t,klist,cnm_list):
        # the trotterlization of both k and t space
        dk = klist[1]-klist[0]
        Vkist = self.V_Cat_t_klist(t, klist,cnm_list)
        return np.sum(Vkist*dk,axis=0)
