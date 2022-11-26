# My Utility : Back-Propagation

import pandas as pd
import numpy  as np   

# Randomized Weight 
def randW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)

# Feed-forward of the ANN
def forward(x,W,n_function):        
    h = act_function(W[0],x,n_function)
    y = act_function(W[1],h, 5)
    return y,h

#Activation function
def act_function(w,X,function_number):
    z = np.dot(w, X.T)
    if(function_number==1):
        h_z = ReLu_function(z)
    if(function_number==2):
        h_z = L_ReLu_function(z)
    if(function_number==3):
        h_z = ELU_function(z)
    if(function_number==4):
        h_z = SELU_function(z)
    if(function_number==5):
        h_z = sigmoidal_function(z).T
    return(h_z)
    
# Derivative of  Activation function    
def derivate_act(z,function_number):
    if(function_number==1):h_z = d_ReLu_function(z)
    elif(function_number==2):h_z = d_L_ReLu_function(z)
    elif(function_number==3):h_z = d_ELU_function(z)
    elif(function_number==4):h_z = d_SELU_function(z)
    elif(function_number==5):h_z = d_sigmoidal_function(z)
    return(h_z)

# STEP 2: Feed-Backward: 
def ann_gradW(x,w,v,h,e,function_number):
    h = h.T
    x = np.asmatrix(x)
    delta_0 = np.multiply(e , derivate_act(np.dot(v,h),5))
    dE_dv = np.dot(delta_0.T, h.T)
    gh = np.asarray(np.dot(v.T, delta_0.T))
    der = np.asarray(derivate_act(np.dot(w,x.T),function_number))
    delta_h = np.multiply(gh , der.T)
    if(function_number == 5):
        dE_dw = np.dot( delta_h, np.asarray(x))
    else:
        dE_dw = np.dot( delta_h, np.asarray(x))

    return dE_dw, dE_dv

# Update Ws
def ann_updW(w,v,g_w,g_v,mu):
    w = w - mu*g_w
    v = v - mu*g_v
    return w,v
    
def output_activation(v,h):
    z = np.dot(v, h.T)
    y = 1/(1+np.exp(-z))
    return y.T

def ReLu_function(x):
    return np.where(x>0,x,0)

def L_ReLu_function(x):
    return np.where(x<0,0.01*x,x)

def ELU_function(x):
    a = 1.6732
    return np.where(x>0,x,a*(np.exp(x)-1))

def SELU_function(x):
    a = 1.6732
    lam =1.0507
    return np.where(x>0,x*lam,a*(np.exp(x)-1))
      
def sigmoidal_function(z):
    return 1.0/(1.0+np.exp(-z))

def d_ReLu_function(x):
    return np.maximum(0,x)

def d_L_ReLu_function(x):
    return np.where(x<0,0.01*x,x)

def d_ELU_function(x):
    a = 1.6732
    return np.where(x>0,1, a*np.exp(x))

def d_SELU_function(x):
    lam = 1.0507; 
    a = 1.6732
    return np.where(x>0, 1, a*np.exp(x))*lam
      
def d_sigmoidal_function(z):
    return (np.multiply(1/(1+np.exp(-z)),1-(1/(1+np.exp(-z))))).T
#-----------------------------------------------------------------------