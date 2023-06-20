import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science','nature'])

basedir = [
            '230103_s05_fashion_noise0.0_seqnet0500_lif_ep20_lr0.001_T120_alpha100_beta1_pf1e3_bn_initxunif',
            '230103_s05_fashion_noise0.0_seqnet0500_neur_ep20_lr0.001_T120_alpha100_beta1_pf1e3_bn_initxunif',
            '230103_s05_fashion_noise0.0_seqnet0500_neurmw_ep20_lr0.001_T120_alpha100_beta5_pf1e3_bn_initxunif',
            ]
lstyle = ['--','-','-',':',':']
col = ['black','tab:blue','tab:orange','tab:blue','tab:orange']
figacc,axacc = plt.subplots(1,1,figsize=(1.8,1.4))
fignorm,axnorm = plt.subplots(1,1,figsize=(1.8,1.4))
for id,i in enumerate(basedir):
    print(i)
    accs = np.load('./outputs/' + i + '/accuracies.npy')
    epochs = np.linspace(0,np.shape(accs)[0]-1,np.shape(accs)[0])+0.5
    # noise_inj = np.load('./outputs/' + i + '/noise_std_range.npy')
    axacc.plot(epochs,np.mean(accs,axis=1),lstyle[id],color=col[id])
    if id < 3:
        axacc.fill_between(epochs,np.mean(accs,axis=1)-np.std(accs,axis=1),np.mean(accs,axis=1)+np.std(accs,axis=1),alpha=0.3,facecolor=col[id])
    # axnorm.plot(np.mean(accs,axis=1)/np.mean(accs,axis=1)[0],'^-')
axacc.set_ylim([79,89])
# axacc.set_yticks([76,79,82,85,88])
figacc.savefig('accs.svg')
# fignorm.savefig('accs_normed.svg')