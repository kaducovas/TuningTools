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
from collections import OrderedDict

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

def avgNestedLists(nested_vals):
  """
  Averages a 2-D array and returns a 1-D array of all of the columns
  averaged together, regardless of their dimensions.
  """
  output = []
  maximum = 0
  for lst in nested_vals:
    if len(lst) > maximum:
      maximum = len(lst)
  for index in range(maximum): # Go through each index of longest list
    temp = []
    for lst in nested_vals: # Go through each list
      if index < len(lst): # If not an index error
        temp.append(lst[index])
    output.append(np.nanmean(temp))
  return output

def stdNestedLists(nested_vals):
  """
  Averages a 2-D array and returns a 1-D array of all of the columns
  averaged together, regardless of their dimensions.
  """
  output = []
  maximum = 0
  for lst in nested_vals:
    if len(lst) > maximum:
      maximum = len(lst)
  for index in range(maximum): # Go through each index of longest list
    temp = []
    for lst in nested_vals: # Go through each list
      if index < len(lst): # If not an index error
        temp.append(lst[index])
    output.append(np.nanstd(temp))
  return output

def plot_AE_training(fname,dirout):
  png_files=[]
  with open(fname) as f:
    content = f.readlines()
  f.close()

  layers_list =[f.split('/')[-1].split('_')[24] for f in content]
  layers=sorted(list(set(layers_list)),cmp=layer2number)
  print(layers)
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
    #print len(loss.values())
    #print list(loss.values())
    max_epochs = np.max(epochs.values())
    loss_mean = avgNestedLists(list(loss.values())) #np.mean(list(loss.values()),axis=0)
    loss_std = stdNestedLists(list(loss.values())) #np.std(loss.values(),axis=0)
    val_loss_mean = avgNestedLists(list(val_loss.values())) #np.mean(val_loss.values(),axis=0)
    val_loss_std = stdNestedLists(list(val_loss.values())) #np.std(val_loss.values(),axis=0)
    kl_mean = avgNestedLists(list(kl.values())) #np.mean(kl.values(),axis=0)
    kl_std = stdNestedLists(list(kl.values())) #np.std(kl.values(),axis=0)
    val_kl_mean = avgNestedLists(list(val_kl.values())) #np.mean(val_kl.values(),axis=0)
    val_kl_std = stdNestedLists(list(val_kl.values())) #np.std(val_kl.values(),axis=0)



    fig, axs = plt.subplots(2, 2, figsize=(24, 18))
    plt.figure(1)
    ##PLOT MSE TREINAMENTO
    #list_t = []
    plt.subplot(221)
    plt.errorbar(range(max_epochs+1),y=loss_mean,yerr=loss_std,errorevery=10)
    #3#for i in range(len(epochs.keys())):
      ###plt.plot(epochs[i],loss[i])
      #plt.plot(max_epochs,loss_mean)
    #plt.plot(T[i].history['val_loss'])
    #list_t.append('Sorteio %.f'%(i+1))
    list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
    print list_t
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

    plt.errorbar(range(max_epochs+1),y=val_loss_mean,yerr=val_loss_std,errorevery=10)
    #for i in range(len(epochs.keys())):
    #  plt.plot(epochs[i],val_loss[i])
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

    plt.errorbar(range(max_epochs+1),y=kl_mean,yerr=kl_std,errorevery=10)
    #for i in range(len(epochs.keys())):
    #  plt.plot(epochs[i],kl[i])
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
    plt.errorbar(range(max_epochs+1),y=val_kl_mean,yerr=val_kl_std,errorevery=10)

    #for i in range(len(epochs.keys())):
    #  plt.plot(epochs[i],val_kl[i])
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

def save_dl_model(path=None,model=None):
  # serialize model to JSON
  model_json = model.to_json()
  with open(path+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(path+".h5")
  #print("Saved model to disk")

def load_dl_model(path=None,model=None):
  # load json and create model
  json_file = open(path+".json", 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(path+".h5")
  print("Loaded model from disk")
  return loaded_model

def print_metrics(metricsDict):
  for key in metricsDict.keys():
    if isinstance(metricsDict[key], float):
      print("{:15}: {:.2f}".format(key, metricsDict[key]))
    else:
      print("{:15}: {}".format(key, metricsDict[key]))

    return 0

def report_performance(labels, predictions, elapsed=0, model_name="",time=None,sort=None,etBinIdx=None,etaBinIdx=None,phase=None,report=True):
  from sklearn.metrics         import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
  import dataset
  db = dataset.connect('sqlite:///:memory:')
  table = db['metrics']
  metrics = OrderedDict()
  predictions[predictions >= 0] = 1
  predictions[predictions < 0] = -1
  metrics['Model'] = model_name
  metrics['time'] = time
  metrics['phase'] = phase
  metrics['Elapsed'] = elapsed
  metrics['sort'] = sort
  metrics['etBinIdx'] = etBinIdx
  metrics['etaBinIdx'] = etaBinIdx
  metrics['accuracy'] = accuracy_score(labels, predictions, normalize=True)
  metrics['f1'] = f1_score(labels, predictions)
  metrics['auc'] = roc_auc_score(labels, predictions)
  metrics['precision'] = precision_score(labels, predictions)
  metrics['recall'] = recall_score(labels, predictions)

  if report == True:
    print_metrics(metrics)

  return metrics

def cross_val_analysis_nn(n_split=10, classifier=None, x=None, y=None, model_name="",
              patience=30, train_verbose=2, n_epochs=500):
  '''
    Classification and ROC analysis
    Run classifier with cross-validation and plot ROC curves
  '''
  kf = KFold(n_splits=n_split)
  kf.get_n_splits(x)

  tprs = []
  fpr_ = []
  tpr_ = []
  aucs = []
  accuracy_ = []
  f1_score_ = []
  precision_ = []
  recall_ = []
  roc_auc_ = []

  metrics_ = {}
  trn_desc = {}
  mean_fpr = np.linspace(0, 1, 100)

  batch_size = min(x[y==-1].shape[0],x[y==1].shape[0])

  i = 0
  #start_time = time.time()
  for train, val in kf.split(x,y):
    print('Train Process for %i Fold'%(i+1))
    #print("TRAIN:", train_index, "TEST:", test_index)
    #trainX, valX = trainDf[train_index], trainDf[val_index]
    #trainY, valY = y_train[train_index], y_train[val_index]

    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=train_verbose, mode='auto')
    model = classifier.fit(x.iloc[train], y[train], nb_epoch=n_epochs, callbacks=[earlyStopping], verbose=train_verbose, validation_data=(x.iloc[val], y[val]))
    trn_desc[i] = model
    #model = classifier.fit(x.iloc[train], y[train])
    pred_ = model.predict(x.iloc[val])
    probas_ = model.predict_proba(x.iloc[val])

    # Metrics evaluation
    accuracy_.append(100*accuracy_score(y[val],pred_ , normalize=True))
    f1_score_.append(100*f1_score(y[val], pred_))
    roc_auc_.append(100*roc_auc_score(y[val], pred_))
    precision_.append(100*precision_score(y[val], pred_))
    recall_.append(100*recall_score(y[val], pred_))


    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[val], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    fpr_.append(fpr)
    tpr_.append(tpr)
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
         label='ROC fold %d (AUC = %0.2f)' % (i, 100*roc_auc))

    i += 1
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
       label='Luck', alpha=.8)

  #Store average and std metrics in dict
  metrics_['model']=model_name
  metrics_['accuracy']=round(np.mean(accuracy_),2)
  metrics_['accuracy_std']=round(np.std(accuracy_),2)
  #metrics_['fpr']=round(np.mean(fpr_),2)
  #metrics_['fpr_std']=round(np.std(fpr_),2)
  #metrics_['tpr']=round(np.mean(tpr_),2)
  #metrics_['tpr_std']=round(np.std(tpr_),2)
  metrics_['precision']=round(np.mean(precision_),2)
  metrics_['precision_std']=round(np.std(precision_),2)
  metrics_['recall']=round(np.mean(recall_),2)
  metrics_['recall_std']=round(np.std(recall_),2)
  metrics_['roc_auc']=round(np.mean(roc_auc_),2)
  metrics_['roc_auc_std']=round(np.std(roc_auc_),2)
  metrics_['f1']=round(np.mean(f1_score_),2)
  metrics_['f1_std']=round(np.std(f1_score_),2)


  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(aucs)
  plt.plot(mean_fpr, mean_tpr, color='b',
       label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (100*mean_auc, 100*std_auc),
       lw=2, alpha=.8)

  std_tpr = np.std(tprs, axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
           label=r'$\pm$ 1 std. dev.')

  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(model_name+' Receiver operating characteristic')
  plt.legend(loc="lower right")
  plt.show()

  return metrics_,trn_desc
