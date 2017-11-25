#!/usr/bin/env python

import numpy as np


etBins       = [3, 7, 10, 15]
#etaBins      = [0, 0.8 , 1.37, 1.54, 2.37, 2.47, 2.5]
etaBins      = [0, 0.8 , 1.37, 1.54, 2.37, 2.47]

# eta bins - from CERN
# eta 0.0, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37 



tight20170713 = np.array(
# eta 0           0.8         1.37       1.54     2.37
  [[0.484,        0.340,      0.272,     0.234,   0.139] # Et 4
  ,[0.511,        0.497,      0.391,     0.304,   0.223] # Et 7
  ,[0.564,        0.509,      0.416,     0.384,   0.262] # Et 10
  ,[0.650,        0.630,      0.584,     0.524,   0.431]])*100. # Et 15


# need to fix this values.
medium20170713 = np.array(
# eta 0          0.8         1.37        1.54    2.37
  [[ 0.906125, 0.8907973,  0.8385,  0.8812,   0.88125263] # Et 4
  ,[ 0.924125, 0.91683784, 0.8438,  0.9121,   0.91210316] # Et 7
  ,[ 0.944885, 0.93741676, 0.84908, 0.9240,   0.92400337] # Et 10
  ,[ 0.947125, 0.94508108, 0.8595,  0.9384,   0.93848421]])*100. # Et 15

loose20170713 = np.array(
# eta 0          0.8         1.37        1.54    2.37
  [[ 0.9425,  0.93227027,  0.876,  0.9196,     0.9196    ] # Et 4
  ,[ 0.95465, 0.94708108, 0.8706,  0.9347,     0.93477684] # Et 7
  ,[ 0.96871, 0.96318919, 0.87894, 0.9518,     0.95187642] # Et 10
  ,[ 0.97525, 0.97298649, 0.887,   0.9670,     0.96703158]])*100. # Et 15
  
veryloose20170713 = np.array(
# eta 0          0.8      1.37     1.54         2.37
  [[ 0.978,   0.96458108, 0.9145,  0.9578,      0.95786316] # Et 4
  ,[ 0.98615, 0.97850541, 0.9028,  0.9673,      0.96738947] # Et 7
  ,[ 0.99369, 0.9900427,  0.90956, 0.9778,      0.97782105] # Et 10
  ,[ 0.99525, 0.99318919, 0.9165,  0.9858,      0.98582632]])*100. # Et 15


mytight = tight20170713 + 0.5*(np.ones_like(tight20170713)*100 - tight20170713)
mymedium = medium20170713 + 0.5*(np.ones_like(medium20170713)*100 - medium20170713)
myloose = loose20170713 + 0.5*(np.ones_like(loose20170713)*100 - loose20170713)
myveryloose = veryloose20170713 + 0.5*(np.ones_like(veryloose20170713)*100 - veryloose20170713)
#etaBins      = [0, 0.8]



#for ref in (veryloose20160701, loose20160701, medium20160701, tight20160701):
ref = tight20170713
from RingerCore import traverse
pdrefs = ref
#print pdrefs
pfrefs = np.array( [[0.05]*len(etaBins)]*len(etBins) )*100. # 3 5 7 10
efficiencyValues = np.array([np.array([refs]) for refs in zip(traverse(pdrefs,tree_types=(np.ndarray),simple_ret=True)
                                                 ,traverse(pfrefs,tree_types=(np.ndarray),simple_ret=True))]).reshape(pdrefs.shape + (2,) )


basePath     = '/home/jodafons/CERN-DATA/data/mc16_13TeV'
sgnInputFile = 'user.jodafons.mc16_13TeV.423200.Pythia8B_A14_CTEQ6L1_Jpsie3e3.Physval.s5005_GLOBAL'
bkgInputFile = 'user.jodafons.mc16_13TeV.361237.Pythia8EvtGen_A3NNPDF23LO_minbias_inelastic.Physval.s5007_GLOBAL'
outputFile   = 'sample'
treePath     = ["*/HLT/Physval/Egamma/probes",
                "*/HLT/Physval/Egamma/fakes"]

import os.path
from TuningTools import Reference, RingerOperation
from TuningTools import createData
from RingerCore  import LoggingLevel
from TuningTools.dataframe import Dataframe

createData( sgnFileList      = os.path.join( basePath, sgnInputFile ),
            bkgFileList      = os.path.join( basePath, bkgInputFile ),
            ringerOperation  = RingerOperation.Trigger,
            referenceSgn     = Reference.Off_Likelihood, # probes passed by vloose
            referenceBkg     = Reference.Off_Likelihood, # electrons/any reproved by very loose
            treePath         = treePath,
            pattern_oFile    = outputFile,
            l2EtCut          = 3,
            offEtCut         = 4,
            #nClusters        = 100,
            etBins           = etBins,
            etaBins          = etaBins,
            toMatlab         = False,
            #efficiencyValues = efficiencyValues,
            plotMeans        = True,
            plotProfiles     = False,
            dataframe        = Dataframe.PhysVal_v2,
            level     = LoggingLevel.VERBOSE
          )



from RingerCore import traverse
from TuningTools import BenchmarkEfficiencyArchieve
 
refname_list= ['veryloose','loose','medium','tight']
pdrefs_list = [myveryloose, myloose, mymedium, mytight]
pfrefs_list = [0.07,0.05,0.03,0.01]


effFile  = outputFile+'-eff.npz'
refFile     = BenchmarkEfficiencyArchieve.load(effFile,False,None,None,True,True)
nEtBins     = refFile.nEtBins
nEtaBins    = refFile.nEtaBins
etBins      = refFile.etBins
etaBins     = refFile.etaBins
sgnEff      = refFile.signalEfficiencies
bkgEff      = refFile.backgroundEfficiencies
sgnCrossEff = refFile.signalCrossEfficiencies
bkgCrossEff = refFile.backgroundCrossEfficiencies
operation   = refFile.operation


for refIdx in range(len(pdrefs_list)):
  refname = refname_list[refIdx]
  pdrefs  = pdrefs_list[refIdx]
  pfrefs  = np.array( [[pfrefs_list[refIdx]]*len(etaBins)]*len(etBins) )*100.
  efficiencyValues = np.array([np.array([refs]) for refs in zip(traverse(pdrefs,tree_types=(np.ndarray),simple_ret=True)
                                                   ,traverse(pfrefs,tree_types=(np.ndarray),simple_ret=True))]).reshape(pdrefs.shape + (2,) )
  #print efficiencyValues

  for etBinIdx in range(nEtBins):
    for etaBinIdx in range(nEtaBins):
      
      for key in sgnEff.keys():
        sgnEff[key][etBinIdx][etaBinIdx].setEfficiency(pdrefs[etBinIdx][etaBinIdx])
        bkgEff[key][etBinIdx][etaBinIdx].setEfficiency(pfrefs[etBinIdx][etaBinIdx])
  
  kwin = {'etaBins':                     etaBins
         ,'etBins':                      etBins
         ,'signalEfficiencies':          sgnEff
         ,'backgroundEfficiencies':      bkgEff
         ,'signalCrossEfficiencies':     sgnCrossEff
         ,'backgroundCrossEfficiencies': bkgCrossEff
         ,'operation':                   operation}
  
  ofile = BenchmarkEfficiencyArchieve(kwin).save(outputFile+'.'+refname+'-eff.npz')




