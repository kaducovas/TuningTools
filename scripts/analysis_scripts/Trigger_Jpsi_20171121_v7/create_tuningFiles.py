

from TuningTools.CreateTuningJobFiles import createTuningJobFiles
createTuningJobFiles( outputFolder   = 'Jpsi.config.n5to20.JK.inits_100by100',
                      neuronBounds   = [5,20],
                      sortBounds     = 10,
                      nInits         = 100,
                      nNeuronsPerJob = 1,
                      nInitsPerJob   = 100,
                      nSortsPerJob   = 1,
                      prefix         = 'Job',
                      compress       = True )



from TuningTools.CrossValid import CrossValid, CrossValidArchieve
crossValid = CrossValid(nSorts = 10,
                        nBoxes = 10,
                        nTrain = 9, 
                        nValid = 1,
                        )
place = CrossValidArchieve( 'crossValid', 
                            crossValid = crossValid,
                            ).save( True )


from TuningTools.PreProc import *
ppCol = PreProcChain( Norm1() ) 
from TuningTools.TuningJob import fixPPCol
ppCol = fixPPCol(ppCol)
place = PreProcArchieve( 'ppFile', ppCol = ppCol ).save()

