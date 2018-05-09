from sklearn.metrics import mutual_info_score
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid.axes_grid import AxesGrid
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from matplotlib.ticker import ScalarFormatter
import scipy
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd
import numpy
import numpy as np
import os
import re
import matplotlib.pyplot as plt

def calc_MI(x, y):
  max_value = max(max(x),max(y))
  min_value = min(min(x),min(y))
  bins = min( len(np.histogram(x,'fd')[0]), len(np.histogram(y,'fd')[0]))
  bins_list = np.linspace(min_value, max_value, num=bins)
  c_xy,xaaa,yaaa = np.histogram2d(x, y, bins=(bins_list,bins_list))
  mi = mutual_info_score(None, None, contingency=c_xy)
  return mi #,xaaa,yaaa,bins
  
def layer2number(x, y):
  return int(y.split('x')[1]) - int(x.split('x')[1])     

def plot_AE_training(fname,dirout):
  png_files=[]
  with open(fname) as f:
    content = f.readlines()
  f.close()
  
  layers_list =[f.split('/')[-1].split('_')[24] for f in content]
  layers=sorted(list(set(layers_list)),cmp=layer2number)

  list_t=[]
  for layer in layers:
    epochs = {}
    loss = {}
    kl = {}
    val_loss = {}
    val_kl = {}
    files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer)]
    for file in files:
      job = joblib.load(file.replace('\n','')+'_trn_desc.jbl')
      file_name = file.split('/')[-1]
      #print file.split('_')[27]
      epochs[int(file_name.split('_')[27])] = job[0]['epochs']
      loss[int(file_name.split('_')[27])] = job[0]['loss']
      kl[int(file_name.split('_')[27])] = job[0]['kullback_leibler_divergence']
      val_loss[int(file_name.split('_')[27])] = job[0]['val_losS']
      val_kl[int(file_name.split('_')[27])] = job[0]['val_kullback_leibler_divergence']
    
    max_epochs = np.max(epochs.values())
    loss_mean = np.mean(loss.values())
    val_loss_mean = np.mean(val_loss.values()) 
    kl_mean = np.mean(kl.values())
    val_kl_mean = np.mean(val_kl.values())



    fig, axs = plt.subplots(2, 2, figsize=(24, 18))
    plt.figure(1)
    ##PLOT MSE TREINAMENTO
    #list_t = []
    plt.subplot(221)
    for i in range(len(epochs.keys())):
      plt.plot(epochs[i],loss[i])
      #plt.plot(max_epochs,loss_mean)
    #plt.plot(T[i].history['val_loss'])
    #list_t.append('Sorteio %.f'%(i+1))
      list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
      plt.legend(list_t, loc='best',fontsize = 'xx-large')
    #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
    #plt.title('SAE - '+layer.replace('x','-')+'-'+layer.split('x')[0],fontsize= 'xx-large')
    plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
    plt.ylabel('Erro de Treinamento (MSE)',fontsize= 'xx-large')
    plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
    plt.tick_params(axis='both',labelsize=16)
    #plt.xlim(0)
    #plt.grid()
    plt.yscale('log')
    
    ##PLOT MSE Val
    #list_t = []
    plt.subplot(222)
    for i in range(len(epochs.keys())):
      plt.plot(epochs[i],val_loss[i])
      #plt.plot(max_epochs,val_loss_mean)
    #plt.plot(T[i].history['val_loss'])
    #list_t.append('Sorteio %.f'%(i+1))
    #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
      plt.legend(list_t, loc='best',fontsize = 'xx-large')
    #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
    plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
    plt.ylabel(r'Erro de $Validac\c{}\~ao$ (MSE)',fontsize= 'xx-large')
    plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
    plt.tick_params(axis='both',labelsize=16)
    #plt.xlim(0)
    #plt.grid()
    plt.yscale('log')
    
    ##PLOT KL TREINAMENTO
    
    #list_t = []
    plt.subplot(223)
    for i in range(len(epochs.keys())):
      plt.plot(epochs[i],kl[i])
      #plt.plot(max_epochs,kl_mean)
    #plt.plot(T[i].history['val_loss'])
    #list_t.append('Sorteio %.f'%(i+1))
    #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
      plt.legend(list_t, loc='best',fontsize = 'xx-large')
    #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
    plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
    plt.ylabel('Erro de Treinamento (KL)',fontsize= 'xx-large')
    plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
    plt.tick_params(axis='both',labelsize=16)
    #plt.xlim(0)
    #plt.grid()
    plt.yscale('log')
    
    ##PLOT KL Val

    #list_t = []
    plt.subplot(224)
    for i in range(len(epochs.keys())):
      plt.plot(epochs[i],val_kl[i])
      #plt.plot(max_epochs,val_kl_mean)
    #plt.plot(T[i].history['val_loss'])
    #list_t.append('Sorteio %.f'%(i+1))
    #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
      plt.legend(list_t, loc='best',fontsize = 'xx-large')
    #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
    plt.title('SAE - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
    plt.ylabel(r'Erro de $Validac\c{}\~ao$ (KL)',fontsize= 'xx-large')
    plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
    plt.tick_params(axis='both',labelsize=16)
    #plt.xlim(0)
    #plt.grid()
    plt.yscale('log')    
  
  #plt.grid()
  plt.savefig(dirout+'layer_'+layer+'_'+fname.split('/')[-1]+'.png')
  png_files.append(dirout+'layer_'+layer+'_'+fname.split('/')[-1]+'.png')
  return png_files
  
    #plt.show()
