
python runGRIDtuning.py \
 -c ~mverissi/jpsi.config.n5to20.JK.inits_100by100 \
 -d ~mverissi/Jpsi_sample.npz \
 -p ~mverissi/ppFile.pic.gz \
 -x ~mverissi/crossValid.pic.gz \
 -r ~mverissi/Jpsi_sample.tight-eff.npz \
 -o user.mverissi.Jpsi.test \
 --eta-bin 0 4 \
 --et-bin 0 4 \
 --excludedSite ANALY_BNL_MCORE \
 --multi-files \
 #-mt \
 #--memory 17010 \
 #--dry-run

