runTuning.py -d /home/wsfreund/CERN-DATA/Offline/tuningData/mc16calo_v1/mc16_13TeV.sgn.truth.bkg.truth.offline.binned.calo.npz \
    --neuronBounds 16 16 \
    --initBounds 0 1 \
    --sortBounds 0 1 \
    --et-bins 2 2 \
    --eta-bins 0 0 \
    --output-level DEBUG \
    --operation Offline_LH_Medium \
