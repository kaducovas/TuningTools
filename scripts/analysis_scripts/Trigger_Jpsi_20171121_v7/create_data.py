#!/usr/bin/env python

import numpy as np
import make_eff

etBins       = [3, 7, 10, 15]
etaBins      = [0, 0.8 , 1.37, 1.54, 2.37, 2.47]

#for ref in (veryloose20160701, loose20160701, medium20160701, tight20160701):
ref = make_eff.transformToEffCalo()
from RingerCore import traverse
pdrefs = ref
#print pdrefs
pfrefs = np.array( [[0.05]*len(etaBins)]*len(etBins) )*100. # 3 5 7 10
efficiencyValues = np.array([np.array([refs]) for refs in zip(traverse(pdrefs,tree_types=(np.ndarray),simple_ret=True)
                                                 ,traverse(pfrefs,tree_types=(np.ndarray),simple_ret=True))]).reshape(pdrefs.shape + (2,) )

basePathCERN = '/afs/cern.ch/user/m/mverissi/private/DATA' 
basePath     = '/home/jodafons/CERN-DATA/data/mc16_13TeV'
sgnInputFile = 'user.jodafons.mc16_13TeV.423200.Pythia8B_A14_CTEQ6L1_Jpsie3e3.Physval.s5005_GLOBAL'
bkgInputFile = 'user.jodafons.mc16_13TeV.361237.Pythia8EvtGen_A3NNPDF23LO_minbias_inelastic.Physval.s5007_GLOBAL'
outputFile   = 'Jpsi_sample'
treePath     = ["*/HLT/Physval/Egamma/probes",
                "*/HLT/Physval/Egamma/fakes"]

import os.path
from TuningTools import Reference, RingerOperation
from TuningTools import createData
from RingerCore  import LoggingLevel
from TuningTools.dataframe import Dataframe

createData( sgnFileList      = os.path.join( basePathCERN, sgnInputFile ),
            bkgFileList      = os.path.join( basePathCERN, bkgInputFile ),
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



