
python runGRIDtuning.py \
 -c user.mverissi.new.v2.sample.Jpsi.20172911.v2.config.n5to20.JK.inits_100by100 \
 -d user.mverissi.new.sample.Jpsi.20172911.npz \
 -pp user.mverissi.new.ppFile.20172911.pic.gz \
 -x user.mverissi.new.crossValid.20172911.pic.gz \
 -r user.mverissi.new.sample.Jpsi.20172911.tight-eff.npz \
 -o user.mverissi.new.output.20173011 \
 --eta-bin 0 4 \
 --et-bin 0 2 \
 --excludedSite ANALY_BNL_MCORE \
 #--multi-files \
 #-mt \
 #--memory 17010 \
 #--dry-run

