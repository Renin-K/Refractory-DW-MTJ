import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science','nature'])

basedir = ['1115_inf_neur','1115_inf_lif']
figacc,axacc = plt.subplots(1,1,figsize=(2,1.4))
fignorm,axnorm = plt.subplots(1,1,figsize=(2,1.4))
for i in basedir:
    accs = np.load('./outputs/' + i + '/accs_np.npy')
    noise_inj = np.load('./outputs/' + i + '/noise_std_range.npy')
    axacc.plot(noise_inj,np.mean(accs,axis=1),'^-')
    axacc.fill_between(noise_inj,np.mean(accs,axis=1)-np.std(accs,axis=1),np.mean(accs,axis=1)+np.std(accs,axis=1),alpha=0.3)
    axnorm.plot(noise_inj,np.mean(accs,axis=1)/np.mean(accs,axis=1)[0],'^-')
figacc.savefig('accs.svg')
fignorm.savefig('accs_normed.svg')