import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science','nature'])

basedir = ['230112_inf_neur','230112_inf_neurmw','230112_inf_lif']
lstyle = ['-','-','--']
col = ['tab:blue','tab:orange','black']
figacc,axacc = plt.subplots(1,1,figsize=(1.8,1.4))
fignorm,axnorm = plt.subplots(1,1,figsize=(1.8,1.4))
for id,i in enumerate(basedir):
    accs = np.load('./outputs/' + i + '/accs_np.npy')
    noise_inj = np.load('./outputs/' + i + '/noise_std_range.npy')
    axacc.plot(noise_inj,np.mean(accs,axis=1),lstyle[id],color=col[id])
    # axacc.fill_between(noise_inj,np.mean(accs,axis=1)-np.std(accs,axis=1),np.mean(accs,axis=1)+np.std(accs,axis=1),facecolor=col[id],alpha=0.3)
    axnorm.plot(noise_inj,np.mean(accs,axis=1)/np.mean(accs,axis=1)[0],lstyle[id],color=col[id])
    # axnorm.fill_between(noise_inj)
figacc.savefig('inf_accs.svg')
fignorm.savefig('inf_accs_normed.svg')