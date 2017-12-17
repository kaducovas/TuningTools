runGRIDtuning.py -d /scratch/22061a/caducovas/mc16_13TeV.sgn.truth.bkg.truth.offline.binned.calo.npz \
    --pp /scratch/22061a/caducovas/ppFile_StackedAutoEncoder.pic \
    -c /scratch/22061a/caducovas/SAE_config \
    -x /scratch/22061a/caducovas/crossValid_SAE_jackknife.pic -o /scratch/22061a/caducovas/caloNN \
    --et-bins 2 2 \
    --eta-bins 0 0 \
    --output-level DEBUG \
    --operation Offline_LH_Medium \
