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
def ann_bp(w1,w2,x,y):
    ...    
    return(w1,w2)

# Training : ANN-PSO
def ann_pso(x,y,particles_amount):
    


    return(w1,w2)
# Training:ANN-PSO and  ANN-BP
def train_ann(x,y,param):
    w1,w2 = ann_pso(x,y,...)
    w1,w2 = ann_bp(w1,w2,x,y,...)    
    return(w1,w2) 
   
# Beginning ...
def main():
    param   = load_config()            
    xe,ye   = load_data()   
    w1,w2   = train_ann(xe,ye,param)             
    save_w(w1,w2)
       
if __name__ == '__main__':   
	 main()