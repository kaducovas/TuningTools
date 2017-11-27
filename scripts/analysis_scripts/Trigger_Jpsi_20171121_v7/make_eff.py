import numpy as np

etBins       = [3, 7, 10, 15]
etaBins      = [0, 0.8, 1.37, 1.54, 2.37, 2.47]

tight20170713 = np.array(

    [[0.484,0.492,0.340,0.316,0.272,0.400,0.384,0.234,0.139], # 4 GeV 
     [0.511,0.579,0.497,0.472,0.391,0.385,0.379,0.304,0.223], # 7 GeV
     [0.564,0.631,0.509,0.492,0.416,0.517,0.510,0.384,0.262], # 10 GeV
     [0.650,0.642,0.630,0.605,0.584,0.607,0.570,0.524,0.431]] # 15 GeV

    )*100.0 

medium20170713 = np.array( 

    [[0.677,0.644,0.532,0.497,0.273,0.539,0.522,0.436,0.324], # 4 GeV 
     [0.690,0.707,0.685,0.649,0.287,0.571,0.553,0.467,0.387], # 7 GeV
     [0.731,0.758,0.666,0.632,0.419,0.641,0.633,0.567,0.479], # 10 GeV
     [0.765,0.754,0.722,0.621,0.637,0.714,0.654,0.642,0.527]] # 15 GeV

     )*100.0

loose20170713 = np.array(
    
    [[0.817,0.824,0.791,0.765,0.556,0.718,0.699,0.690,0.619], # 4 GeV 
     [0.819,0.786,0.773,0.747,0.440,0.748,0.729,0.703,0.631], # 7 GeV
     [0.823,0.850,0.767,0.741,0.512,0.767,0.748,0.736,0.667], # 10 GeV
     [0.846,0.852,0.779,0.768,0.702,0.784,0.775,0.743,0.642]] # 15 GeV

    )*100.0

veryloose20170713 = np.array(
    
    [[0.916,0.893,0.900,0.894,0.789,0.885,0.876,0.839,0.731], # 4 GeV 
     [0.923,0.920,0.922,0.916,0.638,0.876,0.867,0.870,0.784], # 7 GeV
     [0.908,0.935,0.882,0.875,0.696,0.866,0.857,0.809,0.726], # 10 GeV
     [0.948,0.920,0.922,0.885,0.789,0.880,0.885,0.879,0.796]] # 15 GeV

    )*100.0
  

def mergeEffTable(eff = 'tight'):
    if eff == 'tight':
        val = tight20170713
    elif eff == 'medium':
        val = medium20170713
    elif eff == 'loose':
        val = loose20170713
    elif eff == 'veryloose':
        val = veryloose20170713
  
    shorterEtaEffTable = np.zeros(shape=(len(etBins),len(etaBins)))
    for eta_index in range(len(etaBins)):
        for et_index in range(len(etBins)):
            if eta_index == 0:
                # merge 0 -> .6 -> .8
                shorterEtaEffTable[et_index, eta_index] = (val[et_index, 0]*.6 + val[et_index, 1]*.2) / .8
            if eta_index == 1:
                # merge 1.15 -> 1.37 -> 1.52
                shorterEtaEffTable[et_index, eta_index] = (val[et_index, 2]*.22 + val[et_index, 3]*.15) / .37
            if eta_index == 2:
                # 1.37 -> 1.52
                shorterEtaEffTable[et_index, eta_index] = val[et_index, 4]
            if eta_index == 3:
                # merge 1.52 -> 1.8 -> 2.47
                shorterEtaEffTable[et_index,eta_index] = (val[et_index, 5]*.29 + val[et_index, 6]*.2 + val[et_index, 7]*.46) / .95
            else:
                shorterEtaEffTable[et_index,eta_index] = val[et_index,eta_index]
    return shorterEtaEffTable

def transformToEffCalo (eff = 'tight'):   

    if eff == 'tight':
      val = mergeEffTable(eff)

    elif eff == 'medium':
      val = mergeEffTable(eff)

    elif eff == 'loose':
      val = mergeEffTable(eff)

    elif eff == 'veryloose':
      val = mergeEffTable(eff)
    return val + np.minimum(0.5*(np.ones_like(val)*100. - val),val) 




