# My Utility : Algorithm PSO 

import numpy    as np
import pandas   as pd
import util_bp  as bp
import math

def iniSwarm(K,m,L,n_p):
    w =  bp.randW(L,m)
    v = bp.randW(K,L)
    D = (L * m) + (K*L)
    S = []
    P = np.ndarray( shape=(n_p,D) )
    V = np.ndarray( shape=(n_p,D) )
    Pg = np.ndarray( shape = (1,D) )

    for i in range(n_p):
        S.append(np.concatenate((w.reshape(-1),v.reshape(-1)),axis=0))
    S = np.asmatrix(S)
    P[np.isnan(P)] = 0,
    V[np.isnan(V)] = 0
    Pg[np.isnan(Pg)] = 0

    return(S, P, Pg, V)

# Fitness by use MSE
def Fitness_mse(y, x,S, n_p,L):
    m = x.shape[1]
    N = len(y[0])
    K = y.shape[1]
    MSE = []
    for i in range(n_p):
        w = S[i,:m*L].reshape(L,m)
        v = S[i,-K*L:].reshape(K,L)
        h_z = hidden_activation(w,x)
        y_predict = pd.DataFrame(output_activation(v,h_z))
        diff = ((y_predict- y)**2).sum()/N
        MSE.append((diff[0]+diff[1])/2)
   
    return(MSE)
    
# Update:Particles based on Fitness-MSE
def upd_pFitness(P, Pg,Fits,x,y,S,L,best_MSE, n_p):
    P_mse = np.asarray(Fitness_mse(y,x,P,n_p,L))
    MSE = np.asarray(Fits)

    for i in range(n_p):
        if MSE[i] < P_mse[i]: 
            P[i] = S[i]
            if MSE[i] < best_MSE:
                Pg = S[i]
                best_MSE = MSE[i]
    return(P, Pg,best_MSE)
    
# Update: Swarm's velocity
def upd_veloc(S,P,Pg,V,alpha):
    c1 = 1.05
    c2 = 2.95
    r1 = np.random.rand(1)
    r2 = np.random.rand(1)
    new_V = alpha*V + c1*r1*(S-P)+ c2*r2(S-Pg)
    return(new_V)  
  
def hidden_activation(w,X,function_number):
    z = np.matmul(w,X.T)
    h_z = 1/(1+np.exp(-z))
    return(h_z)

def output_activation(v,h):
    z = np.matmul(v, h)
    y = 1/(1+np.exp(-z))
    return y.T

def calc_alpha(i_actual, i_max):
    alpha_max = 0.95
    alpha_min = 0.1
    return (alpha_max -((alpha_max-alpha_min)/i_max)*i_actual)

def ReLu_function(x):
    for i in range(len(x)):
        if x[i] <= 0:
            x[i] = 0
    return x

def L_ReLu_function(x):
    for i in range(len(x)):
        if x[i] <= 0:
            x[i] = 0
        else:
            x[i] = 0.01*x[i]
    return x
def ELU_function(x):
    a = 1.6732
    for i in range(len(x)):
        if x[i] <= 0:
            x[i] = a*(np.exp(x[i])-1)

    return x  

def SELU_function(x):
    a = 1.6732
    lam = 1.0507
    for i in range(len(x)):
        if x[i] <= 0:
            x[i] = lam*a*(np.exp(x[i])-1)
        else:
            x[i] = lam* x[i]
    return x
      
def sigmoidal_function(prev,next):
    z = np.matmul(prev, next)
    sig = 1/(1+np.exp(-z))
    return sig.T
    
#-----------------------------------------------------------------------
