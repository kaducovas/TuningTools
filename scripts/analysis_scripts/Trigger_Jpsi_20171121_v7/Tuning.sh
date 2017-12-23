python runTuning.py \
  -d /home/micael.verissimo/RingerProject/root/TuningTools/scripts/analysis_scripts/Trigger_Jpsi_20171121_v7/sample.Jpsi.20172911.npz\
  --pp /home/micael.verissimo/RingerProject/root/TuningTools/scripts/analysis_scripts/Trigger_Jpsi_20171121_v7/ppFile.20172911.pic.gz \
  -x /home/micael.verissimo/RingerProject/root/TuningTools/scripts/analysis_scripts/Trigger_Jpsi_20171121_v7/crossValid.20172911.pic.gz \
  --neuronBounds 5 5 \
  --do-multi-stop False
  #--outputFileBase tunedDiscr \
  --eta-bins 0 --et-bins 2 \
  --epochs 50000 --output-level DEBUG \
  --development 