#/usr/bin/env python


from RingerCore.Configure import development
print 'import feito'
development = True

from timeit import default_timer as timer
from RingerCore.Logger import Logger, LoggingLevel
from TuningTools.TuningJob import TuningJob
from TuningTools.PreProc import *
import logging

start = timer()
DatasetLocationInput = '/home/micael.verissimo/RingerProject/root/TuningTools/scripts/analysis_scripts/Trigger_Jpsi_20171121_v7/sample.Jpsi.20172911.npz'

ppCol = PreProcChain( RingerEtaMu() ) 
#ppCol = PreProcChain( RingerRp(alpha=0.5,beta=0.5) ) 
from TuningTools.TuningJob import fixPPCol
#ppCol = fixPPCol(ppCol)

#from RingerCore.Configure import development
#development.set(True)


tuningJob = TuningJob()
tuningJob( DatasetLocationInput, 
           neuronBoundsCol = [5, 5], 
           sortBoundsCol = [0, 10],
           initBoundsCol = 100, 
           epochs = 500,
           showEvo = 10,
           doMultiStop = False,
           maxFail = 100,
           #ppCol = ppCol,
           level = 10,
           etBins = 2,
           etaBins = 0,
           crossValidFile='/home/micael.verissimo/RingerProject/root/TuningTools/scripts/analysis_scripts/Trigger_Jpsi_20171121_v7/crossValid.20172911.pic.gz',
           ppFile='/home/micael.verissimo/RingerProject/root/TuningTools/scripts/analysis_scripts/Trigger_Jpsi_20171121_v7/ppFile.20172911.pic.gz',
           development = True
           #confFileList='config.n5to20.jackKnife.inits_100by100/job.hn0009.s0000.il0000.iu0099.pic.gz',
           #refFile='/home/micael.verissimo/RingerProject/root/TuningTools/scripts/analysis_scripts/Trigger_Jpsi_20171121_v7/sample.Jpsi.20172911.tight-eff.npz',
           )


end = timer()

print 'execution time is: ', (end - start)      
