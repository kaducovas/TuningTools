from TuningTools.PreProc import *
ppCol = PreProcChain([Norm1(),StackedAutoEncoder(hidden_neurons=[100]),StackedAutoEncoder(hidden_neurons=[80]),StackedAutoEncoder(hidden_neurons=[60])] )  
from TuningTools.TuningJob import fixPPCol
ppCol = fixPPCol(ppCol)
place = PreProcArchieve( 'teste_SAE_ppFile', ppCol = ppCol ).save()
