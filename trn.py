# Training: IDS ANN-PSO-BP

import pandas   as pd
import numpy    as np
import util_pso as pso
import util_bp  as bp

# Load Parameters
def load_config(file_name:str):
    file_data = np.genfromtxt(file_name, dtype=float, delimiter=",")
    return(file_data)
# Load data of training
def load_data(x_name,y_name):
    x = np.genfromtxt(x_name, dtype=float, delimiter=",")
    y = np.genfromtxt(y_name, dtype=float, delimiter=",")
    return(x,y)
#save weights in numpy format
def save_w():
    ...
    return

# Training: ANN-BP
def ann_bp(w1,w2,x,y, param):
    iteration_number = param[0]
    learning_rate = param[1]
    return(w1,w2)

# Training : ANN-PSO
def ann_pso(x,y,param):
    hidden_activation = param[0]
    hidden_nodes_number = param[1]
    particle_number = param[2]
    iteration_number = param[3]
    space = pso.iniSwarm(iteration_number,particle_number)
    
    return(w1,w2)
# Training:ANN-PSO and  ANN-BP
def train_ann(x,y,param_PSO,param_bp):
    w1,w2 = ann_pso(x,y,param_PSO)
    w1,w2 = ann_bp(w1,w2,x,y,param_bp)    
    return(w1,w2) 
   
# Beginning ...
def main():
    param_PSO = load_config("cnf_ann_pso.csv")            
    param_bp = load_config("cnf_ann_bp.csv")            
    xe,ye = load_data()   
    w1,w2 = train_ann(xe,ye,param_PSO,param_bp)             
    save_w(w1,w2)
       
if __name__ == '__main__':   
	 main()