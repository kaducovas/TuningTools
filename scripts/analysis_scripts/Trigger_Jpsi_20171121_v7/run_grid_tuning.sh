
python runGRIDtuning.py \
 -c user.mverissi.new.v8.sample.Jpsi.20172112.v8.config.n5to20.JK.inits_100by100 \
 -d user.mverissi.new.v3.sample.Jpsi.20172911.npz \
 -pp user.mverissi.new.v4.ppFile.20172112.pic.gz \
 -x user.mverissi.new.v3.crossValid.20172112.pic.gz \
 -r user.mverissi.new.v3.sample.Jpsi.20172911.tight-eff.npz \
 -o user.mverissi.new.output.20172212 \
 --eta-bin 0 4 \
 --et-bin 0 2 \
 --excludedSite ANALY_BNL_MCORE \
 --do-multi-stop False\
 #--multi-files \
 #-mt \
 #--memory 17010 \
 #--dry-run

