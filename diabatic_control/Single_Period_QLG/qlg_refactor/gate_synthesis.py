"""Logical gate synthesis utilities (bosonic codes + QLG control optimization)."""

from __future__ import annotations

import numpy as np
import scipy
from scipy.linalg import expm, eigh, norm
from scipy.optimize import minimize
from joblib import Parallel, delayed

from .qlg import QLG
from .codes import bionomial_state, fourleg_cat_state, finite_gkp_state, displace, get_cnm

def embed_logical_gate(C, U_L):
    """
    输入:
      C   : shape (N,2), 两列分别是逻辑 |0_L>, |1_L> (Fock basis)
      U_L : shape (2,2), 任意逻辑酉
    输出:
      U_phys : shape (N,N), 嵌入物理空间的酉
    """
    # 1) 正交化逻辑基
    G = C.conj().T @ C
    evals, evecs = eigh(G)
    G_inv_sqrt = evecs @ np.diag(evals**-0.5) @ evecs.conj().T
    Q = C @ G_inv_sqrt   # orthonormal logical basis
    P = Q @ Q.conj().T

    # 2) 嵌入
    N = C.shape[0]
    U_phys = Q @ U_L @ Q.conj().T + (np.eye(N) - P)

    # 3) 酉性检查
    err = norm(U_phys.conj().T @ U_phys - np.eye(N))
    if err > 1e-10:
        print("Warning: Unitarity error =", err)
    return U_phys, Q

def global_phase(U: np.ndarray, V: np.ndarray) -> float:
    """
    Find the global phase phi such that V ≈ exp(i*phi) * U.
    Returns phi in (-pi, pi].
    """
    N = U.shape[0]
    overlap = np.trace(U.conj().T @ V) / N
    return np.angle(overlap)

def avg_gate_fidelity(U, V):
    d = U.shape[0]
    t = np.trace(U.conj().T @ V)
    return (np.abs(t)**2 + d) / (d * (d + 1))

def get_H_opt(t,cnm_list,amp,Np):
    # the time-dependent Hamiltonian at time t,amp is the power of the periodic potential
    q_lattice_gate = QLG(Np)
    return  amp*q_lattice_gate.V_Cat_t(t,cnm_list) 

def gate_bosonic_gate_vlist(gate,Np,Nt,delta,states, n_jobs: int = -1):
    '''
        states: the target states
        Np: the Hillbert space
        Nt: Trotter steps
        delta: the bound of for the control functions
        alpha = coherence
        logical = perpare the logical states
    '''
    # ========================input predefined coefficients==============================
    q_lattice_gate = QLG(Np)
    lamd,E ,Dtau,H0,alpha = q_lattice_gate.lamd,q_lattice_gate.E, q_lattice_gate.Dtau, q_lattice_gate.H0, q_lattice_gate.alpha
    if isinstance(states, str):
        if states=='Bio':
            [logical0,logical1] = bionomial_state(Np)
        elif states == 'Cat':
            [logical0,logical1,psicate0,psicate1] = fourleg_cat_state(Np,alpha)
        elif states=='GKP':
            logical0 = finite_gkp_state(Np,0.35,0,8) #sigma, mu, n_range
            logical1 = displace(Np,np.sqrt(np.pi))*logical0
        else:
            raise('wrong states')
    else:
        logical0=states

    #[logical0,logical1] = bionomial_state(Np)
    C =np.stack([logical0.full().T[0],logical1.full().T[0]], axis=1)
    #Qx, Rx = np.linalg.qr(gate)  # QR 得到酉
    U_L = gate#Qx / np.linalg.det(Qx)**0.5  # 调整到 SU(2) # 注意 这里的U_L和gate相差一个global phase
    # 构造物理空间酉
    U_phys, Q = embed_logical_gate(C, U_L)
    # 验证：在逻辑子空间作用等于 U_L
    overlap = Q.conj().T @ U_phys @ Q
    loss =  np.sum(np.abs(np.array(overlap - U_L)))
    global_phi = global_phase(overlap, U_L)
    #print("global phase ≈\n", global_phi)
    #print("Effective logical action ≈\n", overlap)
    #print("Difference from U_L:\n", overlap - U_L, loss)

    #=====================================QLG==============================================#
    U_est = np.exp(1j*global_phi) * U_phys
    H = scipy.linalg.logm(np.array(U_est))*lamd/(-1j*E*Dtau) # the Floquet :e^{-HT/\lamd} = U
    cnm_list = get_cnm(Np,H)
    tlist_finalT = np.linspace(0,Dtau, Nt+1)
    
    Vlist = Parallel(n_jobs=n_jobs,verbose=0)(delayed(get_H_opt)(t,cnm_list,amp=1,Np=Np) for t in tlist_finalT[:])
    return Vlist,Q,cnm_list

def gate_bosonic_gate(gate,Np,Nt,delta,Vlist,Q):
      # ========================input predefined coefficients==============================
    q_lattice_gate = QLG(Np)
    lamd,E ,Dtau,H0,alpha = q_lattice_gate.lamd,q_lattice_gate.E, q_lattice_gate.Dtau, q_lattice_gate.H0, q_lattice_gate.alpha
    tlist_finalT = np.linspace(0,Dtau, Nt+1)
    dt = tlist_finalT[1]-tlist_finalT[0]
    U_L = gate
    def get_qlg_result(Vlist,Np,dt):
        i=0
        h0=H0.full()
        U = np.identity(Np)
        for h in Vlist:
            u = expm(-1j*h*dt/lamd)@expm(-1j*h0*dt/lamd)#@expm(-1j*0.5*h*dt/lamd)
            U = u@U
            i+=1
        overlap = Q.conj().T @ U @ Q
        #FU = np.abs(np.trace(U@U_est.conj().T))/len(U)
        Fgate = avg_gate_fidelity(overlap, U_L)#0.25*np.abs(np.trace(overlap@U_L.conj().T))**2 
        return Fgate,overlap
    
    Fgate_before,gate_before= get_qlg_result(Vlist,Np,dt)
     
    # #======================================QOC=============================================#
    def cost_function(params):
        Vlist1 = [params[i]*Vlist[i] for i in range(len(Vlist))]
        i=0
        h0 = H0.full()
        U = np.identity(Np)
        for h in Vlist1:
            u = expm(-1j*h*dt/lamd)@expm(-1j*h0*dt/lamd)#@expm(-1j*0.5*h*dt/lamd)
            U = u@U
            i+=1
        overlap = Q.conj().T @ U @ Q
        loss = 1-avg_gate_fidelity(overlap, U_L)#1-np.abs(np.trace(U@U_est.conj().T))/len(U)#np.sum(np.abs(np.array(overlap - U_L)))
        return   np.log10(loss)

    #====================================时间参数=========================================
    bounds = [(0, 1+delta)] * len(tlist_finalT)
    initial_guess = np.random.random(len(tlist_finalT))#np.ones_like(tlist_finalT)#
    result = minimize(cost_function, initial_guess, method='SLSQP',bounds=bounds,tol=1e-2)#L-BFGS-B SLSQP L-BFGS-B
    xopt = result.x
    #=======================================================================================
    Vlist1 = [xopt[i]*Vlist[i] for i in range(len(Vlist))]
    Fgate_opt,gate_opt= get_qlg_result(Vlist1,Np,dt)
    return Fgate_before,gate_before,Fgate_opt,gate_opt,xopt 



def gate_bosonic_gate_plot(gate, Np, Nt, delta, Vlist, Q):
    """Small helper used in the notebook: return [F_opt, F_before]."""
    Fgate_before, _gate_before, Fgate_opt, _gate_opt, _xopt = gate_bosonic_gate(gate, Np, Nt, delta, Vlist, Q)
    return [Fgate_opt, Fgate_before]

