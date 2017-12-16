createTuningJobFiles.py all \
              -oJConf SAE_config \
              --nInits 100 \
              --neuronBounds 10 10 \
              -outCross crossValid_SAE_jackknife \
              --method "JackKnife" \
              --nBoxes 10 \
              --nTrain 9 \
              --nSorts 10 \
              -outPP ppFile_StackedAutoEncoder \
              -pp_nEt 5 \
              -pp_nEta 4 \
              -ppCol "[Norm1(),StackedAutoEncoder({'hidden_neurons' : [100]}),StackedAutoEncoder({'hidden_neurons' : [80]}),StackedAutoEncoder({'hidden_neurons' : [60]})]" 
#"[[StackedAutoEncoder({'hidden_neurons' : [80]})]]" #"[Norm1(),StackedAutoEncoder('hidden_neurons' : [80]), StackedAutoEncoder('hidden_neurons' : [60]), StackedAutoEncoder('hidden_neurons' : [40]), StackedAutoEncoder('hidden_neurons' : [20])]"
