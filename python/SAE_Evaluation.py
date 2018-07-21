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
      print file,file.split('_')[27]
      epochs[int(file_name.split('_')[27])] = job[0]['epochs']
      loss[int(file_name.split('_')[27])] = job[0]['loss']
      kl[int(file_name.split('_')[27])] = job[0]['kullback_leibler_divergence']
      val_loss[int(file_name.split('_')[27])] = job[0]['val_losS']
      val_kl[int(file_name.split('_')[27])] = job[0]['val_kullback_leibler_divergence']
    #print len(loss.values())
    #print list(loss.values())
    max_epochs = np.max(epochs.values())
    print 'max_epochs', max_epochs, type(max_epochs)
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
    plt.errorbar(range(len(loss_mean)),y=loss_mean,yerr=loss_std,errorevery=10)
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

    plt.errorbar(range(len(val_loss_mean)),y=val_loss_mean,yerr=val_loss_std,errorevery=10)
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

    plt.errorbar(range(len(kl_mean)),y=kl_mean,yerr=kl_std,errorevery=10)
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
    plt.errorbar(range(len(val_kl_mean)),y=val_kl_mean,yerr=val_kl_std,errorevery=10)

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
  plt.clf()
  plt.close()
  return png_files

def plot_classifier_training(fname,dirout):
  import os
  history_files=[x for x in os.listdir(fname) if x.endswith(".pkl")]
  png_files=[]
  #with open(fname) as f:
  #  content = f.readlines()
  #f.close()

  #layers_list =[f.split('/')[-1].split('_')[24] for f in content]
  #layers=sorted(list(set(layers_list)),cmp=layer2number)
  #print(layers)
  list_t=[]
  #for layer in layers:
  epochs = {}
  loss = {}
  acc = {}
  val_loss = {}
  val_acc = {}
  #files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer)]
  for file in history_files:
    job = load_dl_history(fname+'/'+file) #joblib.load(file.replace('\n','')+'_trn_desc.jbl')
    #print job.keys()
    sort = int(file.split('/')[-1].split('_')[2])
    #print sort
    #print file.split('_')[27]
    epochs[sort] = len(job['loss'])
    loss[sort] = job['loss']
    acc[sort] = job['acc']
    val_loss[sort] = job['val_loss']
    val_acc[sort] = job['val_acc']
  #print len(loss.values())
  #print list(loss.values())
  #max_epochs = np.max(epochs.values())
  #loss_mean = avgNestedLists(list(loss.values())) #np.mean(list(loss.values()),axis=0)
  #loss_std = stdNestedLists(list(loss.values())) #np.std(loss.values(),axis=0)
  #val_loss_mean = avgNestedLists(list(val_loss.values())) #np.mean(val_loss.values(),axis=0)
  #val_loss_std = stdNestedLists(list(val_loss.values())) #np.std(val_loss.values(),axis=0)
  #acc_mean = avgNestedLists(list(acc.values())) #np.mean(kl.values(),axis=0)
  #acc_std = stdNestedLists(list(acc.values())) #np.std(kl.values(),axis=0)
  #val_acc_mean = avgNestedLists(list(val_acc.values())) #np.mean(val_kl.values(),axis=0)
  #val_acc_std = stdNestedLists(list(val_acc.values())) #np.std(val_kl.values(),axis=0)
    #print loss[sort]
  fig, axs = plt.subplots(2, 2, figsize=(24, 18))
  plt.figure(1)
  ##PLOT MSE TREINAMENTO
  #list_t = []
  plt.subplot(221)
  #print loss[0]
  #plt.errorbar(range(max_epochs+1),y=loss_mean,yerr=loss_std,errorevery=10)
  for i in range(len(epochs.keys())):
    #print i,loss[i]
    plt.plot(loss[i])

    #plt.plot(max_epochs,loss_mean)
  #plt.plot(T[i].history['val_loss'])
    list_t.append('Sorteio %.f'%(i+1))
  #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  #print list_t
  plt.legend(list_t, loc='best',fontsize = 'xx-large')
  #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  #plt.title('SAE - '+layer.replace('x','-')+'-'+layer.split('x')[0],fontsize= 'xx-large')
  plt.title('Neural Network - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  plt.ylabel('Erro de Treinamento (MSE)',fontsize= 'xx-large')
  plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  plt.tick_params(axis='both',labelsize=16)
  #plt.xlim(0)
  #plt.grid()
  plt.yscale('log')
  ##PLOT MSE Val
  #list_t = []
  plt.subplot(222)

  #plt.errorbar(range(max_epochs+1),y=val_loss_mean,yerr=val_loss_std,errorevery=10)
  for i in range(len(epochs.keys())):
    plt.plot(val_loss[i])
    #plt.plot(max_epochs,val_loss_mean)
  #plt.plot(T[i].history['val_loss'])
  #list_t.append('Sorteio %.f'%(i+1))
  #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  plt.legend(list_t, loc='best',fontsize = 'xx-large')
  #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  plt.title('Neural Network - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  plt.ylabel(r'Erro de $Validac\c{}\~ao$ (MSE)',fontsize= 'xx-large')
  plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  plt.tick_params(axis='both',labelsize=16)
  #plt.xlim(0)
  #plt.grid()
  plt.yscale('log')

  ##PLOT KL TREINAMENTO

  #list_t = []
  plt.subplot(223)

  #plt.errorbar(range(max_epochs+1),y=kl_mean,yerr=kl_std,errorevery=10)
  for i in range(len(epochs.keys())):
    plt.plot(acc[i])
    #plt.plot(max_epochs,kl_mean)
  #plt.plot(T[i].history['val_loss'])
  #list_t.append('Sorteio %.f'%(i+1))
  #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  plt.legend(list_t, loc='best',fontsize = 'xx-large')
  #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  plt.title('Neural Network - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  plt.ylabel('Erro de Treinamento (Acc)',fontsize= 'xx-large')
  plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  plt.tick_params(axis='both',labelsize=16)
  #plt.xlim(0)
  #plt.grid()
  #plt.yscale('log')
  ##PLOT KL Val

  #list_t = []
  plt.subplot(224)
  #plt.errorbar(range(max_epochs+1),y=val_kl_mean,yerr=val_kl_std,errorevery=10)

  for i in range(len(epochs.keys())):
    plt.plot(val_acc[i])
    #plt.plot(max_epochs,val_kl_mean)
  #plt.plot(T[i].history['val_loss'])
  #list_t.append('Sorteio %.f'%(i+1))
  #list_t.append('AE - '+layer.replace('x','-')+'-'+layer.split('x')[0])
  plt.legend(list_t, loc='best',fontsize = 'xx-large')
  #plt.title('AE '+layer+' - ',fontsize= 'xx-large')
  plt.title('Neural Network - '+fname.split('/')[-1].split('_2018')[0],fontsize= 'xx-large')
  plt.ylabel(r'Erro de $Validac\c{}\~ao$ (Acc)',fontsize= 'xx-large')
  plt.xlabel(r"""$\'Epoca$""",fontsize= 'xx-large')
  plt.tick_params(axis='both',labelsize=16)
  #plt.xlim(0)
  #plt.grid()
  #plt.yscale('log')

  #plt.grid()
  plt.savefig(dirout+'dl_'+fname.split('/')[-1]+'.png')
  png_files.append(dirout+'dl_'+fname.split('/')[-1]+'.png')
  return png_files

def save_dl_history(path,obj):
  import pickle
  with open(path + '.pkl', 'wb') as f:
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
  f.close()

def load_dl_history(path ):
  import pickle
  with open(path, 'rb') as f:
    return pickle.load(f)
  f.close()
def save_dl_model(path=None,model=None):
  # serialize model to JSON
  model_json = model.to_json()
  with open(path+".json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights(path+".h5")
  #print("Saved model to disk")

def load_dl_model(path=None,model=None):
  from keras.models import model_from_json
  import json
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

def report_performance(labels, predictions, elapsed=0, model_name="",hl_neuron=None,time=None,sort=None,etBinIdx=None,etaBinIdx=None,phase=None,point=None,fine_tuning=None,report=True):
  from sklearn.metrics         import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score
  import dataset
  db = dataset.connect('sqlite:////scratch/22061a/caducovas/run/mydatabase.db')
  #print point.sp_value
  table = db['classifier_dm']
  metrics = OrderedDict()
  print len(predictions)
  predictions[predictions >= point.thres_value] = 1
  predictions[predictions < point.thres_value] = -1
  print 'debugging report_performance'
  print labels
  print predictions
  metrics['Model'] = model_name
  metrics['HL_Neuron'] = hl_neuron
  metrics['time'] = time
  metrics['sort'] = sort
  metrics['etBinIdx'] = etBinIdx
  metrics['etaBinIdx'] = etaBinIdx
  metrics['phase'] = phase
  metrics['Elapsed'] = elapsed
  metrics['fine_tuning'] = fine_tuning
  metrics['signal_samples'] = len(labels[labels==1])
  metrics['bkg_samples'] = len(labels[labels==-1])
  metrics['signal_pred_samples'] = len(predictions[predictions==1])
  metrics['bkg_pred_samples'] = len(predictions[predictions==-1])
  metrics['threshold']=float(point.thres_value)
  metrics['sp'] = float(point.sp_value)
  metrics['pd'] = float(point.pd_value)
  metrics['pf'] = float(point.pf_value)
  metrics['accuracy'] = accuracy_score(labels, predictions, normalize=True)
  metrics['f1'] = f1_score(labels, predictions)
  metrics['auc'] = roc_auc_score(labels, predictions)
  metrics['precision'] = precision_score(labels, predictions)
  metrics['recall'] = recall_score(labels, predictions)
  table.insert(metrics)
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

def createClassifierTable(model_name,script_time):
  import dataset
  from prettytable import PrettyTable

  x = PrettyTable()
  x.field_names = ["KPI", "Train", "Validation"]
  db = dataset.connect('sqlite:////scratch/22061a/caducovas/run/mydatabase.db')
  #table = db['classifier']

  #query = 'select model,time,phase, avg(elapsed) as elapsed, avg(signal_samples) as signal_samples,avg(bkg_samples) as bkg_samples,avg(signal_pred_samples) as signal_pred_samples,avg(bkg_pred_samples) as bkg_pred_samples,avg(threshold) as threshold,  avg(sp) || "+-" || stdev(sp) as sp, avg(pd) || "+-" || stdev(pd) as pd, avg(pf) || "+-" || stdev(pf) as pf, avg(accuracy) || "+-" || stdev(accuracy) as accuracy, avg(f1) || "+-" || stdev(f1) as f1, avg(auc) || "+-" || stdev(auc) as auc,  avg(precision) || "+-" || stdev(precision) as precision, avg(recall) || "+-" || stdev(recall) as recall from classifier group by model,time,phase'

  query = 'select model,time,phase,fine_tuning, max(elapsed) as elapsed, avg(signal_samples) as signal_samples,avg(bkg_samples) as bkg_samples,avg(signal_pred_samples) as signal_pred_samples,avg(bkg_pred_samples) as bkg_pred_samples, 100*round(avg(threshold),5) as threshold,  100*round(avg(sp),5) as sp, 100*round(avg(pd),5) as pd, 100*round(avg(pf),5) as pf, 100*round(avg(accuracy),5) as accuracy, 100*round(avg(f1),5) as f1, 100*round(avg(auc),5) as auc, 100*round(avg(precision),5) as precision, 100*round(avg(recall),5) as recall from classifier_dm where model = "'+model_name+'" and time = "'+script_time+'" group by model,time,phase,fine_tuning'
  trnquery = 'select model,time,phase,fine_tuning, max(elapsed) as elapsed, cast(avg(signal_samples) as integer)  as signal_samples,avg(bkg_samples) as bkg_samples,avg(signal_pred_samples) as signal_pred_samples,avg(bkg_pred_samples) as bkg_pred_samples, 100*round(avg(threshold),5) as threshold,  100*round(avg(sp),5) as sp, 100*round(avg(pd),5) as pd, 100*round(avg(pf),5) as pf, 100*round(avg(accuracy),5) as accuracy, 100*round(avg(f1),5) as f1, 100*round(avg(auc),5) as auc, 100*round(avg(precision),5) as precision, 100*round(avg(recall),5) as recall from classifier_dm where model = "'+model_name+'" and time = "'+script_time+'" and phase = "Train" group by model,time,phase,fine_tuning'
  valquery = 'select model,time,phase,fine_tuning, max(elapsed) as elapsed, avg(signal_samples) as signal_samples,avg(bkg_samples) as bkg_samples,avg(signal_pred_samples) as signal_pred_samples,avg(bkg_pred_samples) as bkg_pred_samples, 100*round(avg(threshold),5) as threshold,  100*round(avg(sp),5) as sp, 100*round(avg(pd),5) as pd, 100*round(avg(pf),5) as pf, 100*round(avg(accuracy),5) as accuracy, 100*round(avg(f1),5) as f1, 100*round(avg(auc),5) as auc, 100*round(avg(precision),5) as precision, 100*round(avg(recall),5) as recall from classifier_dm where model = "'+model_name+'" and time = "'+script_time+'" and phase = "Validation" group by model,time,phase,fine_tuning'

  result = db.query(trnquery)
  trnresult = db.query(trnquery)
  valresult = db.query(valquery)
  for row in result:
    chave= row.keys()

  for row in trnresult:
    trn= row #.values()

  for row in valresult:
    val= row #.values()

  for k in chave:
    x.add_row([k,trn[k],val[k]])

  return x

def plot_Roc(fname,dirout, model_name=""):
  import os
  from RingerCore import load
  from sklearn.metrics import roc_curve, auc
  history_files=[x for x in os.listdir(fname) if x.endswith(".pic")]
  png_files=[]

  fig, axs = plt.subplots(1, 1)
  plt.figure(1)

  list_t=[]
  #fig, axs = plt.subplots(1, 2, figsize=(24, 18))
  #fig, axs = plt.subplots(1, 1, figsize=(24, 18))
  #plt.figure(1)
  #@@plt.subplot(121)
  #@@for idx,file in enumerate(history_files):
    #@@disc=load(fname+'/'+file)
    #files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer)]
    #@@pds=disc['tunedDiscr'][0][0]['summaryInfo']['roc_operation']['pds']
    #@@pfs=disc['tunedDiscr'][0][0]['summaryInfo']['roc_operation']['pfs']
    #@@sps=disc['tunedDiscr'][0][0]['summaryInfo']['roc_operation']['sps']
    #@@idxSP = np.argmax(sps)
    #@@sp=sps[idxSP]
    #@@roc_auc = auc(pfs,pds)
    #@@plt.plot(pfs, pds,label='ROC - AUC = '+str(round(roc_auc,4))+', SP = '+str(100*round(sp,4))+' - Sorteio '+str(idx+1)+' ' % roc_auc)
  #@@plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
  #@@plt.xlim([0.0, 1.0])
  #@@plt.ylim([0.0, 1.05])
  #@@plt.xlabel('Probabilidade de Falso Positivo',fontsize= 'xx-large')
  #@@plt.ylabel(r'Probabilidade de $Detec\c{}\~ao$ ',fontsize= 'xx-large')
  #@@plt.title(model_name+' Curva ROC - Treino',fontsize= 'xx-large')
  #@@plt.legend(loc="lower right")

  #plt.subplot(111)
  for idx,file in enumerate(history_files):
    disc=load(fname+'/'+file)
    #files = [f for f in content if (f.split('/')[-1].split('_')[24] == layer)]
    pds=disc['tunedDiscr'][0][0]['summaryInfo']['roc_test']['pds']
    pfs=disc['tunedDiscr'][0][0]['summaryInfo']['roc_test']['pfs']
    sps=disc['tunedDiscr'][0][0]['summaryInfo']['roc_test']['sps']
    idxSP = np.argmax(sps)
    sp=sps[idxSP]
    roc_auc = auc(pfs,pds)
    plt.plot(pfs, pds,label='ROC - AUC = '+str(round(roc_auc,4))+', SP = '+str(100*round(sp,4))+' - Sorteio '+str(idx+1)+' ' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('Probabilidade de Falso Positivo',fontsize= 'xx-large')
  plt.ylabel(r'Probabilidade de $Detec\c{}\~ao$ ',fontsize= 'xx-large')
  plt.title(model_name+' Curva ROC - Teste',fontsize= 'xx-large')
  plt.legend(loc="lower right")

  plt.savefig(dirout+'roc_'+fname.split('/')[-1]+'.png')
  png_files.append(dirout+'roc_'+fname.split('/')[-1]+'.png')
  return png_files
