# Training: IDS ANN-PSO-BP

import pandas   as pd
import numpy    as np
import util_pso as pso
import util_bp  as bp

# Load Parameters
def load_config(pso_str:str,bp_str:str):
    pso = np.genfromtxt(pso_str, dtype=int, delimiter=",")
    bp = np.genfromtxt(bp_str, dtype=float, delimiter=",")
    return(pso,bp)

# Load data of training
def load_data(x_name,y_name):
    x = np.genfromtxt(x_name, dtype=float, delimiter=",")
    y = np.genfromtxt(y_name, dtype=float, delimiter=",")
    return(x,y)

#save weights in numpy format
def save_w(w1,w2):
    W = [w1,w2]
    np.savez('pesos.npz', W=W)
    return

# Training: ANN-BP
def ann_bp(w,v,x,y,pso_param,bp_param):
    max_iter =int(bp_param[0])
    learning_rate = bp_param[1]
    n_activation = pso_param[0]
    N = x.shape[0]
    weights = [w,v]
    fits =[]
    for i in range(max_iter):
        d,h = bp.forward(x,weights,n_activation)
        diff = d-y
        e = np.sum((np.asarray(diff[:,0]))**2+(np.asarray(diff[:,1]))**2)/(N*2)

        g_w,g_v = bp.ann_gradW(x,w,v,h,diff,n_activation)
        w, v = bp.ann_updW(w, v, g_w,g_v, learning_rate)
        weights=[w, v]
        fits.append(e)
    return (w,v)

# Training : ANN-PSO
def ann_pso(x,y,param): 
    hidden_activation = param[0]
    hidden_nodes_number = param[1]
    particle_number = param[2]
    max_iter = param[3]
    S, P, Pg, V = pso.iniSwarm( L = hidden_nodes_number , m = x.shape[1] ,K = len(y[0]), n_p = particle_number )
    best_mse=[]
    for i in range(max_iter):
        
        Fits = pso.Fitness_mse(y, x, S, particle_number,hidden_nodes_number,hidden_activation)
        P, Pg,good_mse = pso.upd_pFitness(P, Pg, Fits,S,particle_number)
        alpha = pso.calc_alpha(i,max_iter)
        V = pso.upd_veloc(S,P,Pg,V,alpha)
        best_mse.append(good_mse)
        S = S + V
    np.savetxt("mse.csv",best_mse,delimiter=",",fmt="%1.5f")
    Pg = Pg[0]
    w = Pg[:,:x.shape[1]*hidden_nodes_number].reshape(hidden_nodes_number, x.shape[1])
    v = Pg[:,-len(y[0])*hidden_nodes_number:].reshape(len(y[0]),hidden_nodes_number)
    return(w,v)

# Training:ANN-PSO and  ANN-BP
def train_ann(x,y,param_PSO,param_bp):
    w1,w2 = ann_pso(x,y,param_PSO)
    w1,w2 = ann_bp(w1,w2,x,y,param_PSO, param_bp)    
    return(w1,w2) 
   
# Beginning ...
def main():
    param_PSO, param_bp = load_config("cnf_ann_pso.csv","cnf_ann_bp.csv")            
    xe,ye = load_data("xtrn.csv","ytrn.csv")   
    w1,w2 = train_ann(xe,ye,param_PSO,param_bp)             
    save_w(w1,w2)
       
if __name__ == '__main__':   
	 main()