import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_DOWN

plt.style.use('dark_background')
basedir = '230620_temp'

num_sweep = 7
epoch_num = 4

linestyle_str = ['-', ':','--','-.','-', ':','--','-.'] 

w1 = np.linspace(50e-9,25e-9,7)
w2 = np.linspace(50e-9,75e-9,7)

epochs = np.linspace(1,epoch_num,epoch_num)
figacc,axacc = plt.subplots(1,1,figsize=(10,10.4))

accsa = []
accs = np.load('./outputs/'+ basedir +'/fin_acc.npy')

for i in range(num_sweep):
    accsa.append(np.zeros((1,num_sweep)))
    accsa[i] = accs[0,(i)*epoch_num:(i+1)*epoch_num].reshape(1,epoch_num)

    axacc.plot(epochs,np.mean(accsa[i],axis=0),str(linestyle_str[i]), label = str(w2[i]/w1[i]))
    axacc.fill_between(epochs,np.mean(accsa[i],axis=0)-np.std(accsa[i],axis=1),np.mean(accsa[i],axis=0)+np.std(accsa[i],axis=1),alpha=0.3)
axacc.legend(loc='upper left')
axacc.set_title("Accuracy vs. Epochs for Width Ratios")
figacc.savefig('accs.svg')