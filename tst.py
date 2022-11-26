import numpy   as np
import util_bp as bp

# Test: Load data 
def load_data():
    x = np.genfromtxt('xtst.csv', dtype=np.float32,delimiter=",")
    y = np.genfromtxt('ytst.csv', dtype=np.float32,delimiter=",")
    W = np.load('pesos.npz', allow_pickle=True)
    w1,w2 = W['W']
    return(x,y,w1,w2)
#Normalized data

#Measure
def metricas(x:np.ndarray,y:np.ndarray):
    cm = np.zeros((y.shape[1], x.shape[1]))
    for real, predicted in zip(y, x):
        cm[np.argmax(real)][np.argmax(predicted)] += 1
    f_score = []
    for index, feature in enumerate(cm): 
        TP = feature[index]
        FP = cm.sum(axis=0)[index] - TP
        FN = cm.sum(axis=1)[index] - TP
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f_score.append((2 * (precision * recall) / (precision + recall)))
    f_score.append(np.array(f_score).mean())
    np.savetxt('cmatriz.csv', cm)
    np.savetxt('fscores.csv', f_score,fmt="%1.25f")
    return (cm, np.array(f_score))


def main():            
    xv,yv,w,v = load_data()    
    zv,h        = bp.forward(xv,[w,v],5)              
    metricas(yv,zv)     

# Beginning ...
if __name__ == '__main__':   
     main()