import os
import glob

########################################################################################

cmd_add_container = "add_container.sh -f {FILE} --dataset {DSNAME}"
cmd_add_rule = [
                  "rucio add-rule --lifetime \"$((30243600))\" {DATASET}/ 1 BNL-OSG2_SCRATCHDISK",
                  "rucio add-rule --lifetime \"$((30243600))\" {DATASET}/ 1 CERN-PROD_SCRATCHDISK",
                  "rucio add-rule --lifetime \"$((30243600))\" {DATASET}/ 1 DESY-HH_SCRATCHDISK",
                  "rucio add-rule --lifetime \"$((30243600))\" {DATASET}/ 1 MWT2_UC_SCRATCHDISK",
                  "rucio add-rule --lifetime \"$((30243600))\" {DATASET}/ 1 FZK-LCG2_SCRATCHDISK",
                ]
cmd_list_rules = "rucio list-rules {DATASET}"

def add_container(f,ds):
  print cmd_add_container.format(FILE=f,DSNAME=ds)
  os.system(cmd_add_container.format(FILE=f,DSNAME=ds))

def add_rules(ds):
  for cmd in cmd_add_rule:
    print cmd.format(DATASET=ds)
    os.system(cmd.format(DATASET=ds))
  os.system(cmd_list_rules.format(DATASET=ds))



########################################################################################

effFiles = [
			'Jpsi_sample.tight-eff.npz',
			'Jpsi_sample.medium-eff.npz',
			'Jpsi_sample.loose-eff.npz',
			'Jpsi_sample.veryloose-eff.npz'	
					]

patternsFiles = [
			'Jpsi_sample.npz'			
				]

configFileDS = 'Jpsi.config.n5to20.JK.inits_100by100'

crossFile = 'crossValid.pic.gz'

ppFile = 'ppFile.pic.gz'


########################################################################################


#add_container( ppFile, 'user.mverissi.'+ppFile)
#add_rules( 'user.mverissi.'+ppFile)

#add_container( crossFile, 'user.mverissi.'+crossFile)
#add_rules( 'user.mverissi.'+crossFile)

#for ds in effFiles:
#  add_container( ds, 'user.mverissi.'+ds)
#  add_rules( 'user.mverissi.'+ds)

#files = glob.glob(configFileDS+'/*')
#for idx, f in enumerate(files):
#  print 'Attaching ',idx+1,'/',len(files)
#  add_container( f, 'user.mverissi.'+configFileDS )
#add_rules( 'user.mverissi.'+configFileDS)


for ds in patternsFiles:
  add_container( ds, 'user.mverissi.'+ds)
  add_rules( 'user.mverissi.'+ds)


#add_container( 'ppFile_mu0_norm1.pic.gz' , 'user.jodafons.ppFile_mu0_norm1.pic.gz' )
#add_container( 'ppFile_mu25_norm1.pic.gz', 'user.jodafons.ppFile_mu25_norm1.pic.gz' )
#add_container( 'ppFile_mu40_norm1.pic.gz', 'user.jodafons.ppFile_mu40_norm1.pic.gz' )
#add_container( 'ppFile_mu55_norm1.pic.gz', 'user.jodafons.ppFile_mu55_norm1.pic.gz' )

#add_rules( 'user.jodafons.ppFile_mu0_norm1.pic.gz'  )
#add_rules( 'user.jodafons.ppFile_mu25_norm1.pic.gz' )
#add_rules( 'user.jodafons.ppFile_mu40_norm1.pic.gz' )
#add_rules( 'user.jodafons.ppFile_mu55_norm1.pic.gz')

