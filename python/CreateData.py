from RingerCore.Logger import Logger
from RingerCore.util   import checkForUnusedVars, reshape
from RingerCore.FileIO import save, load
from TuningTools.coreDef import retrieve_npConstants
npCurrent, _ = retrieve_npConstants()
import numpy as np

# FIXME This should be integrated into a class so that save could check if it
# is one instance of this base class and use its save method
class TuningDataArchieve( Logger ):
  """
  Context manager for Tuning Data archives

  Version 3: - added eta/et bins compatibility
             - added benchmark efficiency information
             - improved fortran/C integration
             - loads only the indicated bins to memory
  Version 2: - started fotran/C order integration
  Version 1: - save compressed npz file
             - removed target information: classes are flaged as
               signal_rings/background_rings
  Version 0: - save pickle file with numpy data
  """

  _type = np.array('TuningData', dtype='|S10')
  _version = np.array(3)

  def __init__(self, filePath = None, **kw):
    """
    Either specify the file path where the file should be read or the data
    which should be appended to it:

    with TuningDataArchieve("/path/to/file", 
                           [eta_bin = None],
                           [et_bin = None]) as data:
      data['signal_rings'] # access rings from signal dataset 
      data['background_rings'] # access rings from background dataset
      data['benchmark_effs'] # access benchmark efficiencies

    When setting eta_bin or et_bin to None, the function will return data and
    efficiency for all bins instead of the just one selected.

    TuningDataArchieve( signal_rings = np.array(...),
                       background_rings = np.array(...),
                       eta_bins = np.array(...),
                       et_bins = np.array(...),
                       benchmark_effs = np.array(...), )
    """
    # Both
    Logger.__init__(self, kw)
    self._filePath                      = filePath
    # Saving
    self._signal_rings                  = kw.pop( 'signal_rings',                  npCurrent.fp_array([]) )
    self._background_rings              = kw.pop( 'background_rings',              npCurrent.fp_array([]) )
    self._eta_bins                      = kw.pop( 'eta_bins',                      npCurrent.fp_array([]) )
    self._et_bins                       = kw.pop( 'et_bins',                       npCurrent.fp_array([]) )
    self._signal_efficiencies           = kw.pop( 'signal_efficiencies',           None                   )
    self._background_efficiencies       = kw.pop( 'background_efficiencies',       None                   )
    self._signal_cross_efficiencies     = kw.pop( 'signal_cross_efficiencies',     None                   )
    self._background_cross_efficiencies = kw.pop( 'background_cross_efficiencies', None                   )
    self._toMatlab                      = kw.pop( 'toMatlab',                      False                  )
    # Loading
    self._eta_bin                       = kw.pop( 'eta_bin',                       None                   )
    self._et_bin                        = kw.pop( 'et_bin',                        None                   )
    checkForUnusedVars( kw, self._logger.warning )
    # Make some checks:
    if type(self._signal_rings) != type(self._background_rings):
      raise TypeError("Signal and background types do not match.")
    if type(self._signal_rings) == list:
      if len(self._signal_rings) != len(self._background_rings) \
          or len(self._signal_rings[0]) != len(self._background_rings[0]):
        raise ValueError("Signal and background rings lenghts do not match.")
    if type(self._eta_bins) is list: self._eta_bins=npCurrent.fp_array(self._eta_bins)
    if type(self._et_bins) is list: self._et_bins=npCurrent.fp_array(self._eta_bins)
    if self._eta_bins.size == 1 or self._eta_bins.size == 1:
      raise ValueError("Eta or et bins size are 1.")

  @property
  def filePath( self ):
    return self._filePath

  @property
  def signal_rings( self ):
    return self._signal_rings

  @property
  def background_rings( self ):
    return self._background_rings

  def getData( self ):
    kw_dict =  {'type': self._type,
             'version': self._version,
            'eta_bins': self._eta_bins,
             'et_bins': self._et_bins }
    max_eta = self.__retrieve_max_bin(self._eta_bins)
    max_et = self.__retrieve_max_bin(self._et_bins)
    # Handle rings:
    if max_eta is None and max_et is None:
      kw_dict['signal_rings'] = self._signal_rings
      kw_dict['background_rings'] = self._background_rings
    else:
      if max_eta is None: max_eta = 0
      if max_et is None: max_et = 0
      for et_bin in range( max_et + 1 ):
        for eta_bin in range( max_eta + 1 ):
          bin_str = self.__get_bin_str(et_bin, eta_bin) 
          sgn_key = 'signal_rings_' + bin_str
          kw_dict[sgn_key] = self._signal_rings[et_bin][eta_bin]
          bkg_key = 'background_rings_' + bin_str
          kw_dict[bkg_key] = self._background_rings[et_bin][eta_bin]
        # eta loop
      # et loop
    # Handle efficiencies
    from copy import deepcopy
    kw_dict['signal_efficiencies']           = deepcopy(self._signal_efficiencies)
    kw_dict['background_efficiencies']       = deepcopy(self._background_efficiencies)
    kw_dict['signal_cross_efficiencies']     = deepcopy(self._signal_cross_efficiencies)
    kw_dict['background_cross_efficiencies'] = deepcopy(self._background_cross_efficiencies)
    def efficiency_to_raw(d):
      from RingerCore.util import traverse
      for key, val in d.iteritems():
        for cData, idx, parent, _, _ in traverse(val):
          if parent is None:
            d[key] = cData.toRawObj()
          else:
            parent[idx] = cData.toRawObj()
    if self._signal_efficiencies and self._background_efficiencies:
      efficiency_to_raw(kw_dict['signal_efficiencies'])
      efficiency_to_raw(kw_dict['background_efficiencies'])
    if self._signal_cross_efficiencies and self._background_cross_efficiencies:
      efficiency_to_raw(kw_dict['signal_cross_efficiencies'])
      efficiency_to_raw(kw_dict['background_cross_efficiencies'])
    return kw_dict
  # end of getData


  def _toMatlabDump(self, data):
    import scipy.io as sio
    import pprint
    crossval = None
    kw_dict_aux = dict()

    # Retrieve efficiecies
    for key_eff in ['signal_','background_']:# sgn and bkg efficiencies
      key_eff+='efficiencies'
      kw_dict_aux[key_eff] = dict()
      for key_trigger in data[key_eff].keys():# Trigger levels
        kw_dict_aux[key_eff][key_trigger] = list()
        etbin = 0; etabin = 0
        for obj  in data[key_eff][key_trigger]: #Et
          kw_dict_aux[key_eff][key_trigger].append(list())
          for obj_  in obj: # Eta
            obj_dict = dict()
            obj_dict['count']  = obj_['_count'] if obj_.has_key('_count') else 0
            obj_dict['passed'] = obj_['_passed'] if obj_.has_key('_passed') else 0
            if obj_dict['count'] > 0:
              obj_dict['efficiency'] = obj_dict['passed']/float(obj_dict['count']) * 100
            else:
              obj_dict['efficiency'] = 0
            obj_dict['branch'] = obj_['_branch']
            kw_dict_aux[key_eff][key_trigger][etbin].append(obj_dict)
          etbin+=1 

    # Retrieve rings
    for key in data.keys():
      if 'rings' in key:
        kw_dict_aux[key] = data[key]

    # Retrieve crossval
    crossVal = data['signal_cross_efficiencies']['L2CaloAccept'][0][0]['_crossVal']
    kw_dict_aux['crossVal'] = {
                                'nBoxes'          : crossVal['_nBoxes'],
                                'nSorts'          : crossVal['_nSorts'],
                                'nTrain'          : crossVal['_nTrain'],
                                'nTest'           : crossVal['_nTest'],
                                'nValid'          : crossVal['_nValid'],
                                'sort_boxes_list' : crossVal['_sort_boxes_list'],
                                }

    self._logger.info( 'Saving data to matlab...')
    sio.savemat(self._filePath+'.mat', kw_dict_aux)
  #end of matlabDump


  def save(self):
    self._logger.info( 'Saving data using following numpy flags: %r', npCurrent)
    data = self.getData()
    if self._toMatlab:  self._toMatlabDump(data)
    return save(data, self._filePath, protocol = 'savez_compressed')



  def __enter__(self):
    data = {'et_bins' : npCurrent.fp_array([]),
            'eta_bins' : npCurrent.fp_array([]),
            'signal_rings' : npCurrent.fp_array([]),
            'background_rings' : npCurrent.fp_array([]),
            'signal_efficiencies' : {},
            'background_efficiencies' : {},
            'signal_efficiencies' : {},
            'background_efficiencies' : {},
            }
    npData = load( self._filePath )
    try:
      if type(npData) is np.ndarray:
        # Legacy type:
        data = reshape( npData[0] ) 
        target = reshape( npData[1] ) 
        self._signal_rings, self._background_rings = TuningDataArchieve.__separateClasses( data, target )
        data = {'signal_rings' : self._signal_rings, 
                'background_rings' : self._background_rings}
      elif type(npData) is np.lib.npyio.NpzFile:
        if npData['type'] != self._type:
          raise RuntimeError("Input file is not of TuningData type!")
        # Retrieve bins information, if any
        if npData['version'] == np.array(3): # self._version:
          eta_bins = npData['eta_bins'] if 'eta_bins' in npData else \
                     npCurrent.array([])
          et_bins  = npData['et_bins'] if 'et_bins' in npData else \
                     npCurrent.array([])
          self.__check_bins(eta_bins, et_bins)
          max_eta = self.__retrieve_max_bin(eta_bins)
          max_et = self.__retrieve_max_bin(et_bins)
          if self._eta_bin == self._et_bin == None:
            data['eta_bins'] = npCurrent.fp_array(eta_bins) if max_eta else npCurrent.fp_array([])
            data['et_bins'] = npCurrent.fp_array(et_bins) if max_et else npCurrent.fp_array([])
          else:
            data['eta_bins'] = npCurrent.fp_array([eta_bins[self._eta_bin],eta_bins[self._eta_bin+1]]) if max_eta else npCurrent.fp_array([])
            data['et_bins'] = npCurrent.fp_array([et_bins[self._et_bin],et_bins[self._et_bin+1]]) if max_et else npCurrent.fp_array([])
        # Retrieve data (and efficiencies):
        from TuningTools.FilterEvents import BranchEffCollector, BranchCrossEffCollector
        def retrieve_raw_efficiency(d, et_bins = None, eta_bins = None, cl = BranchEffCollector):
          from RingerCore.util import traverse
          if d is not None:
            if type(d) is np.ndarray:
              d = d.item()
            for key, val in d.iteritems():
              if et_bins is None or eta_bins is None:
                for cData, idx, parent, _, _ in traverse(val):
                  if parent is None:
                    d[key] = cl.fromRawObj(cData)
                  else:
                    parent[idx] = cl.fromRawObj(cData)
              else:
                if type(et_bins) == type(eta_bins) == list:
                  d[key] = []
                  for cEtBin, et_bin in enumerate(et_bins):
                    d[key].append([])
                    for eta_bin in eta_bins:
                      d[key][cEtBin].append(cl.fromRawObj(val[et_bin][eta_bin]))
                else:
                  d[key] = cl.fromRawObj(val[et_bins][eta_bins])
          return d
        if npData['version'] == np.array(3): # self._version:
          if self._eta_bin is None and max_eta is not None:
            self._eta_bin = range( max_eta + 1 )
          if self._et_bin is None and max_et is not None:
            self._et_bin = range( max_et + 1)
          if self._et_bin is None and self._eta_bin is None:
            data['signal_rings'] = npData['signal_rings']
            data['background_rings'] = npData['background_rings']
            try:
              data['signal_efficiencies']           = retrieve_raw_efficiency(npData['signal_efficiencies'])
              data['background_efficiencies']       = retrieve_raw_efficiency(npData['background_efficiencies'])
            except KeyError:
              pass
            try:
              data['signal_cross_efficiencies']     = retrieve_raw_efficiency(npData['signal_cross_efficiencies'], BranchCrossEffCollector)
              data['background_cross_efficiencies'] = retrieve_raw_efficiency(npData['background_cross_efficiencies'], BranchCrossEffCollector)
            except KeyError:
              pass
          else:
            if self._eta_bin is None: self._eta_bin = 0
            if self._et_bin is None: self._et_bin = 0
            if type(self._eta_bin) == type(self._eta_bin) != list:
              bin_str = self.__get_bin_str(self._et_bin, self._eta_bin) 
              sgn_key = 'signal_rings_' + bin_str
              bkg_key = 'background_rings_' + bin_str
              data['signal_rings']                  = npData[sgn_key]
              data['background_rings']              = npData[bkg_key]
              try:
                data['signal_efficiencies']           = retrieve_raw_efficiency(npData['signal_efficiencies'], 
                                                                                self._et_bin, self._eta_bin)
                data['background_efficiencies']       = retrieve_raw_efficiency(npData['background_efficiencies'],
                                                                                self._et_bin, self._eta_bin)
              except KeyError:
                pass
              try:
                data['signal_cross_efficiencies']     = retrieve_raw_efficiency(npData['signal_cross_efficiencies'],
                                                                                self._et_bin, self._eta_bin, BranchCrossEffCollector)
                data['background_cross_efficiencies'] = retrieve_raw_efficiency(npData['background_cross_efficiencies'],
                                                                                self._et_bin, self._eta_bin, BranchCrossEffCollector)
              except KeyError:
                pass
            else:
              if not type(self._eta_bin) is list:
                self._eta_bin = [self._eta_bin]
              if not type(self._et_bin) is list:
                self._et_bin = [self._et_bin]
              sgn_list = []
              bkg_list = []
              for et_bin in self._et_bin:
                sgn_local_list = []
                bkg_local_list = []
                for eta_bin in self._eta_bin:
                  bin_str = self.__get_bin_str(et_bin, eta_bin) 
                  sgn_key = 'signal_rings_' + bin_str
                  sgn_local_list.append(npData[sgn_key])
                  bkg_key = 'background_rings_' + bin_str
                  bkg_local_list.append(npData[bkg_key])
                # Finished looping on eta
                sgn_list.append(sgn_local_list)
                bkg_list.append(bkg_local_list)
              # Finished retrieving data
              data['signal_rings'] = sgn_list
              data['background_rings'] = bkg_list
              indexes = self._eta_bin[:]; indexes.append(self._eta_bin[-1]+1)
              data['eta_bins'] = eta_bins[indexes]
              indexes = self._et_bin[:]; indexes.append(self._et_bin[-1]+1)
              data['et_bins'] = et_bins[indexes]
              try:
                data['signal_efficiencies']           = retrieve_raw_efficiency(npData['signal_efficiencies'], 
                                                                                self._et_bin, self._eta_bin)
                data['background_efficiencies']       = retrieve_raw_efficiency(npData['background_efficiencies'], 
                                                                                self._et_bin, self._eta_bin)
              except KeyError:
                pass
              try:
                data['signal_cross_efficiencies']     = retrieve_raw_efficiency(npData['signal_cross_efficiencies'], 
                                                                                self._et_bin, self._eta_bin, 
                                                                                BranchCrossEffCollector)
                data['background_cross_efficiencies'] = retrieve_raw_efficiency(npData['background_cross_efficiencies'], 
                                                                                self._et_bin, self._eta_bin, 
                                                                                BranchCrossEffCollector)
              except KeyError:
                pass
        elif npData['version'] <= np.array(2): # self._version:
          data['signal_rings']     = npData['signal_rings']
          data['background_rings'] = npData['background_rings']
        else:
          raise RuntimeError("Unknown file version!")
      elif isinstance(npData, dict) and 'type' in npData:
        raise RuntimeError("Attempted to read archive of type: %s_v%d" % (npData['type'],
                                                                          npData['version']))
      else:
        raise RuntimeError("Object on file is of unkown type.")
    except RuntimeError, e:
      raise RuntimeError(("Couldn't read TuningDataArchieve('%s'): Reason:"
          "\n\t %s" % (self._filePath,e,)))
    data['eta_bins'] = npCurrent.fix_fp_array(data['eta_bins'])
    data['et_bins'] = npCurrent.fix_fp_array(data['et_bins'])
    # Now that data is defined, check if numpy information fits with the
    # information representation we need:
    from RingerCore.util import traverse
    if type(data['signal_rings']) is list:
      for cData, idx, parent, _, _ in traverse(data['signal_rings'], (list,tuple,np.ndarray), 2):
        cData = npCurrent.fix_fp_array(cData)
        parent[idx] = cData
      for cData, idx, parent, _, _ in traverse(data['background_rings'], (list,tuple,np.ndarray), 2):
        cData = npCurrent.fix_fp_array(cData)
        parent[idx] = cData
    else:
      data['signal_rings'] = npCurrent.fix_fp_array(data['signal_rings'])
      data['background_rings'] = npCurrent.fix_fp_array(data['background_rings'])
    return data
    
  def __exit__(self, exc_type, exc_value, traceback):
    pass

  def nEtBins(self):
    """
      Return maximum eta bin index. If variable is not dependent on bin, return none.
    """
    et_max = self.__max_bin('et_bins') 
    return et_max + 1 if et_max is not None else et_max

  def nEtaBins(self):
    """
      Return maximum eta bin index. If variable is not dependent on bin, return none.
    """
    eta_max = self.__max_bin('eta_bins')
    return eta_max + 1 if eta_max is not None else eta_max

  def __max_bin(self, var):
    """
      Return maximum dependent bin index. If variable is not dependent on bin, return none.
    """
    npData = load( self._filePath )
    try:
      if type(npData) is np.ndarray:
        return None
      elif type(npData) is np.lib.npyio.NpzFile:
        if npData['type'] != self._type:
          raise RuntimeError("Input file is not of TuningData type!")
        arr  = npData[var] if var in npData else npCurrent.array([])
        return self.__retrieve_max_bin(arr)
    except RuntimeError, e:
      raise RuntimeError(("Couldn't read TuningDataArchieve('%s'): Reason:"
          "\n\t %s" % (self._filePath,e,)))

  def __retrieve_max_bin(self, arr):
    """
    Return  maximum dependent bin index. If variable is not dependent, return None.
    """
    max_size = arr.size - 2
    return max_size if max_size >= 0 else None

  def __check_bins(self, eta_bins, et_bins):
    """
    Check if self._eta_bin and self._et_bin are ok, through otherwise.
    """
    max_eta = self.__retrieve_max_bin(eta_bins)
    max_et = self.__retrieve_max_bin(et_bins)
    # Check if eta/et bin requested can be retrieved.
    errmsg = ""
    if self._eta_bin > max_eta:
      errmsg += "Cannot retrieve eta_bin(%d) from eta_bins (%r). %s" % (self._eta_bin, eta_bins, 
          ('Max bin is: ' + str(max_eta) + '. ') if max_eta is not None else ' Cannot use eta bins.')
    if self._et_bin > max_et:
      errmsg += "Cannot retrieve et_bin(%d) from et_bins (%r). %s" % (self._et_bin, et_bins,
          ('Max bin is: ' + str(max_et) + '. ') if max_et is not None else ' Cannot use E_T bins. ')
    if errmsg:
      raise ValueError(errmsg)

  def __get_bin_str(self, et_bin, eta_bin):
    return 'etBin_' + str(et_bin) + '_etaBin_' + str(eta_bin)

  @classmethod
  def __separateClasses( cls, data, target ):
    """
    Function for dealing with legacy data.
    """
    sgn = data[np.where(target==1)]
    bkg = data[np.where(target==-1)]
    return sgn, bkg


class CreateData(Logger):

  def __init__( self, logger = None ):
    Logger.__init__( self, logger = logger )
    from TuningTools.FilterEvents import filterEvents
    self._filter = filterEvents

  def __call__(self, sgnFileList, bkgFileList, ringerOperation, **kw):
    """
      Creates a numpy file ntuple with rings and its targets
      Arguments:
        - sgnFileList: A python list or a comma separated list of the root files
            containing the TuningTool TTree for the signal dataset
        - bkgFileList: A python list or a comma separated list of the root files
            containing the TuningTool TTree for the background dataset
        - ringerOperation: Set Operation type to be used by the filter
      Optional arguments:
        - output ['tuningData']: Name for the output file
        - referenceSgn [Reference.Truth]: Filter reference for signal dataset
        - referenceBkg [Reference.Truth]: Filter reference for background dataset
        - treePath [Set using operation]: set tree name on file, this may be set to
          use different sources then the default.
            Default for:
              o Offline: Offline/Egamma/Ntuple/electron
              o L2: Trigger/HLT/Egamma/TPNtuple/e24_medium_L1EM18VH
        - efficiencyTreePath [None]: Sets tree path for retrieving efficiency
              benchmarks.
            When not set, uses treePath as tree.
        - nClusters [None]: Number of clusters to export. If set to None, export
            full PhysVal information.
        - getRatesOnly [False]: Do not create data, but retrieve the efficiency
            for benchmark on the chosen operation.
        - etBins [None]: E_T bins where the data should be segmented
        - etaBins [None]: eta bins where the data should be segmented
        - ringConfig [100]: A list containing the number of rings available in the data
          for each eta bin.
        - crossVal [None]: Whether to measure benchmark efficiency splitting it
          by the crossVal-validation datasets
    """
    from TuningTools.FilterEvents import FilterType, Reference, Dataset, BranchCrossEffCollector
    output             = kw.pop('output',             'tuningData'    )
    referenceSgn       = kw.pop('referenceSgn',       Reference.Truth )
    referenceBkg       = kw.pop('referenceBkg',       Reference.Truth )
    treePath           = kw.pop('treePath',           None            )
    efficiencyTreePath = kw.pop('efficiencyTreePath', None            )
    l1EmClusCut        = kw.pop('l1EmClusCut',        None            )
    l2EtCut            = kw.pop('l2EtCut',            None            )
    efEtCut            = kw.pop('efEtCut',            None            )
    offEtCut           = kw.pop('offEtCut',           None            )
    nClusters          = kw.pop('nClusters',          None            )
    getRatesOnly       = kw.pop('getRatesOnly',       False           )
    etBins             = kw.pop('etBins',             None            )
    etaBins            = kw.pop('etaBins',            None            )
    ringConfig         = kw.pop('ringConfig',         None            )
    crossVal           = kw.pop('crossVal',           None            )
    toMatlab           = kw.pop('toMatlab',           False           )
    if 'level' in kw: 
      self.level = kw.pop('level') # log output level
      self._filter.level = self.level
    checkForUnusedVars( kw, self._logger.warning )
    # Make some checks:
    if ringConfig is None:
      ringConfig = [100]*(len(etaBins)-1) if etaBins else [100]
    if type(treePath) is not list:
      treePath = [treePath]
    if type(efficiencyTreePath) is not list:
      efficiencyTreePath = [efficiencyTreePath]
    if len(treePath) == 1:
      treePath.append( treePath[0] )
    if len(efficiencyTreePath) == 1:
      efficiencyTreePath.append( efficiencyTreePath[0] )
    if etaBins is None: etaBins = npCurrent.fp_array([])
    if etBins is None: etBins = npCurrent.fp_array([])
    if type(etaBins) is list: etaBins=npCurrent.fp_array(etaBins)
    if type(etBins) is list: etBins=npCurrent.fp_array(etBins)

    nEtBins  = len(etBins)-1 if not etBins is None else 1
    nEtaBins = len(etaBins)-1 if not etaBins is None else 1
    #useBins = True if nEtBins > 1 or nEtaBins > 1 else False

    #FIXME: problems to only one bin. print eff doest work as well
    useBins=True

    self._logger.info('Extracting signal dataset information...')

    # List of operation arguments to be propagated
    kwargs = { 'l1EmClusCut':  l1EmClusCut,
               'l2EtCut':      l2EtCut,
               'efEtCut':      efEtCut,
               'offEtCut':     offEtCut,
               'nClusters':    nClusters,
               'getRatesOnly': getRatesOnly,
               'etBins':       etBins,
               'etaBins':      etaBins,
               'ringConfig':   ringConfig,
               'crossVal':     crossVal, }

    npSgn, sgnEff, sgnCrossEff  = self._filter(sgnFileList,
                                               ringerOperation,
                                               filterType = FilterType.Signal,
                                               reference = referenceSgn,
                                               treePath = treePath[0],
                                               efficiencyTreePath = efficiencyTreePath[0],
                                               **kwargs)
    if npSgn.size: self.__printShapes(npSgn,'Signal')

    self._logger.info('Extracting background dataset information...')
    npBkg, bkgEff, bkgCrossEff = self._filter(bkgFileList, 
                                              ringerOperation,
                                              filterType = FilterType.Background,
                                              reference = referenceBkg,
                                              treePath = treePath[1],
                                              efficiencyTreePath = efficiencyTreePath[1],
                                              **kwargs)
    if npBkg.size: self.__printShapes(npBkg,'Background')

    if not getRatesOnly:
      savedPath = TuningDataArchieve(output,
                                     signal_rings = npSgn,
                                     background_rings = npBkg,
                                     eta_bins = etaBins,
                                     et_bins = etBins,
                                     signal_efficiencies = sgnEff,
                                     background_efficiencies = bkgEff,
                                     signal_cross_efficiencies = sgnCrossEff,
                                     background_cross_efficiencies = bkgCrossEff,
                                     toMatlab = toMatlab,
                                     ).save()
      self._logger.info('Saved data file at path: %s', savedPath )

    for key in sgnEff.iterkeys():
      for etBin in range(nEtBins):
        for etaBin in range(nEtaBins):
          sgnEffBranch = sgnEff[key][etBin][etaBin] if useBins else sgnEff[key]
          bkgEffBranch = bkgEff[key][etBin][etaBin] if useBins else bkgEff[key]
          self._logger.info('Efficiency for %s: Det(%%): %s | FA(%%): %s', 
                            sgnEffBranch.printName,
                            sgnEffBranch.eff_str(),
                            bkgEffBranch.eff_str() )
          if crossVal is not None:
            for ds in BranchCrossEffCollector.dsList:
              try:
                sgnEffBranchCross = sgnCrossEff[key][etBin][etaBin] if useBins else sgnEff[key]
                bkgEffBranchCross = bkgCrossEff[key][etBin][etaBin] if useBins else bkgEff[key]
                self._logger.info( '%s_%s: Det(%%): %s | FA(%%): %s',
                                  Dataset.tostring(ds),
                                  sgnEffBranchCross.printName,
                                  sgnEffBranchCross.eff_str(ds),
                                  bkgEffBranchCross.eff_str(ds))
              except KeyError, e:
                pass
        # for eff
      # for eta
    # for et
  # end __call__

  def __printShapes(self, npArray, name):
    "Print numpy shapes"
    if not npArray.dtype.type is np.object_:
      self._logger.info('Extracted %s rings with size: %r',name, (npArray.shape))
    else:
      shape = npArray.shape
      for etBin in range(shape[0]):
        for etaBin in range(shape[1]):
          self._logger.info('Extracted %s rings (et=%d,eta=%d) with size: %r', 
                            name, 
                            etBin,
                            etaBin,
                            (npArray[etBin][etaBin].shape if npArray[etBin][etaBin] is not None else ("None")))
        # etaBin
      # etBin

createData = CreateData()

