
import numpy   as np
import util_bp as bp
import pandas as pd

# Test: Load data 
def load_config(file):
    cnf_ann_pso = pd.read_csv(file,header=None)
    return(cnf_ann_pso)

    
def load_data():
    x = np.genfromtxt('xtst.csv', dtype=np.float32, delimiter=',')
    y = np.genfromtxt('ytst.csv', dtype=np.float32, delimiter=',')
    W = np.load('pesos.npz', allow_pickle=True)
    w1,w2 = W['W']
    return(x,y,w1,w2)

#metricas
def metricas(x,y):
    mc = np.zeros(shape=(y.shape[1], x.shape[1]))
    for r, p in zip(y, x):
        mc[np.argmax(r)][np.argmax(p)] += 1
    fscore = []
    for i, value in enumerate(mc):
        TP = value[i]
        FP = mc.sum(axis=0)[i] - TP
        FN = mc.sum(axis=1)[i] - TP
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        fscore.append((2 * (precision * recall) / (precision + recall)))

    fscore.append(np.array(fscore).mean())
    pd.DataFrame(fscore).to_csv('fscores.csv',index=False,header=False)
    pd.DataFrame(mc).astype(int).to_csv('cmatriz.csv',index=False,header=False)
    return (fscore)

def main():			
    params = load_config('cnf_ann_pso.csv')
    n_activation = params[0][0]  
    xv,yv,w1,w2 = load_data()
    zv,h = bp.forward(xv,[w1,w2],n_activation)      		
    metricas(yv,zv) 	

# Beginning ...
if __name__ == '__main__':   
    main()