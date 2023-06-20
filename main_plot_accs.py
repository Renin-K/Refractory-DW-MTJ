import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['science','nature'])

basedir = ['1026_s05_fashion_seqnet0500_neur_ep20_lr1e-3_T60_alpha100_beta25_pf1e3_noclip_nodecay_bn_initxunif1.0',
            '1026_s05_fashion_seqnet0500_lif_ep20_lr1e-3_T60_alpha100_beta25_pf1e3_noclip_nodecay_bn_initxunif1.0']
epochs = np.linspace(1,20,20)
figacc,axacc = plt.subplots(1,1,figsize=(2,1.4))
figloss,axloss = plt.subplots(1,1,figsize=(2,1.4))
for i in basedir:
    accs = np.load('./outputs/' + i + '/accuracies.npy')
    loss = np.load('./outputs/' + i + '/test_losses.npy')
    axacc.plot(epochs,np.mean(accs,axis=1),'^-')
    axacc.fill_between(epochs,np.mean(accs,axis=1)-np.std(accs,axis=1),np.mean(accs,axis=1)+np.std(accs,axis=1),alpha=0.3)
    axloss.plot(epochs,np.mean(loss,axis=1),'^-')
    axloss.fill_between(epochs,np.mean(loss,axis=1)-np.std(loss,axis=1),np.mean(loss,axis=1)+np.std(loss,axis=1),alpha=0.3)
figacc.savefig('accs.svg')
figloss.savefig('loss.svg')