# My Utility : Algorithm PSO 

import numpy    as np
import pandas   as pd
import util_bp  as bp

def iniSwarm(K,m,L,n_p):
    
    D = (L * m) + (K*L)
    S = []
    P_position = np.ndarray( shape=(n_p,D) )
    V = np.ndarray( shape=(n_p,D) )
    Pg_position = np.ndarray( shape = (1,D) )

    for i in range(n_p):
        w =  bp.randW(L,m)
        v = bp.randW(K,L)
        S.append(np.concatenate((w.reshape(-1),v.reshape(-1)),axis=0))
    S = np.asmatrix(S)
    P_position[np.isnan(P_position)] = 0,
    V[np.isnan(V)] = 0
    Pg_position[np.isnan(Pg_position)] = 0
    P = [P_position, np.full(shape=P_position.shape[0],fill_value=10^8,dtype=float)]
    Pg = [Pg_position, np.full(shape=Pg_position.shape[0],fill_value=10^8,dtype=float)]
    return(S, P, Pg, V)

# Fitness by use MSE
def Fitness_mse(y, x,S, n_p,L,n_activation):
    m = x.shape[1]
    K = y.shape[1]
    N = x.shape[0]

    fits = np.zeros(n_p)
    for i in range(n_p):
        w = S[i,:m*L].reshape(L,m)
        v = S[i,-K*L:].reshape(K,L)
        weights = [w,v]
        d,h = bp.forward(x,weights,n_activation)
        diff = y-d
        e = np.sum((np.asarray(diff[:,0]))**2+(np.asarray(diff[:,1]))**2)/(N*2)
        fits[i] = e
    return(fits)
    
# Update:Particles based on Fitness-MSE
def upd_pFitness(P, Pg, fits,S, n_p,):
    p_position = P[0]
    p_mse = P[1]
    pg_position = Pg[0]
    pg_mse = Pg[1]

    #print(np.where( fits < p_mse,S,P ))
    for i in range(n_p):
        if(fits[i] < p_mse[i]):
            p_mse[i] = fits[i]
            p_position[i,:] = S[i,:]
            if(p_mse[i] < pg_mse):
                pg_mse = p_mse[i]
                pg_position = S[i,:]
    P = [p_position,p_mse]
    Pg = [pg_position,pg_mse]
    
    return(P,Pg,pg_mse)

# Update: Swarm's velocity
def upd_veloc(S,P,Pg,V,alpha):
    c1 = 1.05
    c2 = 2.95
    r1 = float(np.random.rand(1))
    r2 = float(np.random.rand(1))
    new_V = alpha*V + c1*r1*(P[0]-S) + c2*r2*(Pg[0]-S)
    return(new_V)  


def calc_alpha(i_actual, i_max):
    alpha_max = 0.95
    alpha_min = 0.1
    return (alpha_max -((alpha_max-alpha_min)/i_max)*i_actual)
#-----------------------------------------------------------------------
