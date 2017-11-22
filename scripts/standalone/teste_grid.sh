runGRIDtuning.py -d /home/wsfreund/CERN-DATA/Offline/tuningData/mc16calo_v1/mc16_13TeV.sgn.truth.bkg.truth.offline.binned.calo.npz \
    --pp /home/caducovas/RingerProject/root/TuningTools/scripts/standalone/ppFile_StackedAutoEncoder.pic \
    -c /home/caducovas/RingerProject/root/TuningTools/scripts/standalone/SAE_config \
    -x /home/caducovas/RingerProject/root/TuningTools/scripts/standalone/crossValid_SAE_jackknife.pic \ 
    -o /home/caducovas/RingerProject/root/TuningTools/scripts/standalone/caloNN
    --et-bins 2 2 \
    --eta-bins 0 0 \
    --output-level DEBUG \
    --operation Offline_LH_Medium \
