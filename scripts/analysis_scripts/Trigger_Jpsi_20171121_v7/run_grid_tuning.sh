
python runGRIDtuning.py \
 -c user.mverissi.Jpsi.config.n5to20.JK.inits_100by100 \
 -d user.mverissi.Jpsi_sample_v2.npz \
 -pp user.mverissi.ppFile.pic.gz \
 -x user.mverissi.crossValid.pic.gz \
 -r user.mverissi.Jpsi_sample.tight-eff.npz \
 -o user.mverissi.Jpsi.test \
 --eta-bin 0 4 \
 --et-bin 0 4 \
 --excludedSite ANALY_BNL_MCORE \
 #--multi-files \
 #-mt \
 #--memory 17010 \
 #--dry-run

