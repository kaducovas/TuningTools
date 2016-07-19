__all__ = ['createDataParser','CreateDataNamespace']

from RingerCore import argparse, get_attributes, BooleanStr, \
                       NotSet, LoggerNamespace

from TuningTools.ReadData import RingerOperation

###############################################################################
# Create data related objects
###############################################################################
createDataParser = argparse.ArgumentParser(add_help = False, 
                                           description = 'Create TuningTool data from PhysVal.')
from TuningTools.ReadData import Reference, Detector
mainCreateData = createDataParser.add_argument_group( "Required arguments", "")
mainCreateData.add_argument('-s','--sgnInputFiles', action='store', 
    metavar='SignalInputFiles', required = True, nargs='+',
    help = "The signal files that will be used to tune the discriminators")
mainCreateData.add_argument('-b','--bkgInputFiles', action='store', 
    metavar='BackgroundInputFiles', required = True, nargs='+',
    help = "The background files that will be used to tune the discriminators")
mainCreateData.add_argument('-op','--operation', default = NotSet, 
                     help = """The Ringer operation determining in each Trigger 
                     level or what is the offline operation point reference.
                     Possible options are: """ \
                     + str(get_attributes( RingerOperation, onlyVars = True, getProtected = False)) )
mainCreateData.add_argument('-t','--treePath', metavar='TreePath', action = 'store', 
    default = NotSet, type=str, nargs='+',
    help = """The Tree path to be filtered on the files. It can be a value for
    each dataset.""")
optCreateData = createDataParser.add_argument_group( "Configuration extra arguments", "")
optCreateData.add_argument('--reference', action='store', nargs='+',
    default = NotSet,
    help = """
      The reference used for filtering datasets. It needs to be set
      to a value on the Reference enumeration on ReadData file.
      You can set only one value to be used for both datasets, or one
      value first for the Signal dataset and the second for the Background
      dataset.
         Possible options are: """ \
          + str( get_attributes( Reference, onlyVars = True, getProtected = False) ),
          )
optCreateData.add_argument('-tEff','--efficiencyTreePath', metavar='EfficienciyTreePath', action = 'store', 
    default = NotSet, type=str, nargs='+',
    help = """The Tree path to calculate efficiency. 
    If not specified, efficiency is calculated upon treePath.""")
optCreateData.add_argument('-l1','--l1EmClusCut', default = NotSet, 
    type=float, help = "The L1 cut threshold")
optCreateData.add_argument('-l2','--l2EtCut', default = NotSet, 
    type=float, help = "The L2 Et cut threshold")
optCreateData.add_argument('-ef','--efEtCut', default = NotSet, 
    type=float, help = "The EF Et cut threshold")
optCreateData.add_argument('-off','--offEtCut', default = NotSet, 
    type=float, help = "The Offline Et cut threshold")
optCreateData.add_argument('--getRatesOnly', default = NotSet, 
    action='store_true', help = """Don't save output file, just print benchmark 
                                   algorithm operation reference.""")
optCreateData.add_argument('--etBins', action='store', nargs='+',
    default = NotSet, type=float,
    help = "E_T bins (GeV) where the data should be segmented.")
optCreateData.add_argument('--etaBins', action='store', nargs='+',
    default = NotSet, type=float,
    help = "eta bins where the data should be segmented.")
optCreateData.add_argument('--ringConfig', action='store', nargs='+',
    type=int, default = NotSet, 
    help = "Number of rings for each eta bin segmentation.")
optCreateData.add_argument('-nC','--nClusters', 
    default = NotSet, type=int,
    help = "Maximum number of events to add to each dataset.")
optCreateData.add_argument('-o','--pattern-output-file', default = NotSet, 
    help = "The pickle intermediate file that will be used to train the datasets. It also contains the efficiency targets.")
optCreateData.add_argument('-of','--efficiency-output-file', default = NotSet, 
    help = "File only containing the dumped efficiencies for posterior use.")
optCreateData.add_argument('--crossFile', 
    default = NotSet, type=str,
    help = """Cross-Validation file which will be used to tune the Ringer
    Discriminators.""")
optCreateData.add_argument('--extractDet', action='store', 
    default = NotSet, choices = get_attributes( Detector, onlyVars = True, getProtected = False),
    help = """ Which detector to export data from. """)
optCreateData.add_argument('--standardCaloVariables', default=NotSet, dest = '_standardCaloVariables',
    help = "Whether to use standard calorimeter variables or rings information. Allowed options: " + \
       str( get_attributes( BooleanStr, onlyVars = True, getProtected = False ) )
       )
optCreateData.add_argument('--useTRT', default=NotSet, dest = '_useTRT',
    help = "Enable or disable TRT usage when exporting tracking information. Allowed options: " + \
       str( get_attributes( BooleanStr, onlyVars = True, getProtected = False ) )
       )
################################################################################

################################################################################
# Use this namespace when parsing grid CrossValidStat options
class CreateDataNamespace(LoggerNamespace):
  """
    Parse CrossValidStat options.
  """

  def __init__(self, **kw):
    LoggerNamespace.__init__( self, **kw )

  @property
  def useTRT(self):
    return BooleanStr.treatVar('_useTRT', self.__dict__, False)

  @property
  def standardCaloVariables(self):
    return BooleanStr.treatVar('_standardCaloVariables', self.__dict__, False)

