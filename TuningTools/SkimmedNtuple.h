#ifndef TUNINGTOOLS_SKIMMEDNTUPLE_H
#define TUNINGTOOLS_SKIMMEDNTUPLE_H

#include <TROOT.h>

// Header file for the classes stored in the TTree if any.
#include "vector"

class SkimmedNtuple {
public :
   // Declaration of leaf types
   Int_t           EventNumber;
   Int_t           RunNumber;
   Int_t           RandomRunNumber;
   Int_t           MCChannelNumber;
   Int_t           RandomLumiBlockNumber;
   Int_t           Nvtx;
   Float_t         actualIntPerXing;
   Float_t         averageIntPerXing;
   Float_t         MCPileupWeight;
   Float_t         VertexZPosition;
   Float_t         Zcand_M;
   Float_t         Zcand_pt;
   Float_t         Zcand_eta;
   Float_t         Zcand_phi;
   Float_t         Zcand_y;
   Int_t           isTagTag;
   Int_t           elCand1_isEMLoose;
   Int_t           elCand1_isEMMedium;
   Int_t           elCand1_isEMTight;
   Int_t           elCand1_isTightPP;
   Int_t           elCand1_isMediumPP;
   Int_t           elCand1_isLoosePP;
   Int_t           elCand1_isVeryLooseLL;
   Int_t           elCand1_isLooseLL;
   Int_t           elCand1_isMediumLL;
   Int_t           elCand1_isTightLL;
   Int_t           elCand1_isVeryTightLL;
   Float_t         elCand1_CFT;
   Int_t           elCand1_passCFT;
   Int_t           elCand1_isVeryLooseLL2015;
   Int_t           elCand1_isLooseLL2015;
   Int_t           elCand1_isMediumLL2015;
   Int_t           elCand1_isTightLL2015;
   Int_t           elCand1_isVeryLooseLL2015_CutD0DphiDeta;
   Int_t           elCand1_isLooseLL2015_CutD0DphiDeta;
   Int_t           elCand1_isMediumLL2015_CutD0DphiDeta;
   Int_t           elCand1_isTightLL2015_CutD0DphiDeta;
   Int_t           elCand1_isLoose2015;
   Int_t           elCand1_isMedium2015;
   Int_t           elCand1_isTight2015;
   Int_t           elCand1_isVeryLooseLL2015_v8;
   Int_t           elCand1_isLooseLL2015_v8;
   Int_t           elCand1_isLooseAndBLayerLL2015_v8;
   Int_t           elCand1_isMediumLL2015_v8;
   Int_t           elCand1_isTightLL2015_v8;
   Int_t           elCand1_isTightLLHCaloMC14Truth;
   Int_t           elCand1_isMediumLLHCaloMC14Truth;
   Int_t           elCand1_isLooseLLHCaloMC14Truth;
   Int_t           elCand1_isVeryLooseLLHCaloMC14Truth;
   Int_t           elCand1_isTightLLHCaloMC14;
   Int_t           elCand1_isMediumLLHCaloMC14;
   Int_t           elCand1_isLooseLLHCaloMC14;
   Int_t           elCand1_isVeryLooseLLHCaloMC14;
   Int_t           elCand1_isTightLLHMC15Calo_v8;
   Int_t           elCand1_isMediumLLHMC15Calo_v8;
   Int_t           elCand1_isLooseLLHMC15Calo_v8;
   Int_t           elCand1_isVeryLooseLLHMC15Calo_v8;
   Int_t           elCand1_isTightLLHMC15_v8;
   Int_t           elCand1_isMediumLLHMC15_v8;
   Int_t           elCand1_isLooseLLHMC15_v8;
   Int_t           elCand1_isLooseAndBLayerLLHMC15_v8;
   Int_t           elCand1_isVeryLooseLLHMC15_v8;
   Int_t           elCand1_isTightLL2015_v10;
   Int_t           elCand1_isVeryLooseLL2016_v11;
   Int_t           elCand1_isLooseLL2016_v11;
   Int_t           elCand1_isLooseAndBLayerLL2016_v11;
   Int_t           elCand1_isMediumLL2016_v11;
   Int_t           elCand1_isTightLL2016_v11;
   Int_t           elCand1_TightLLH_80GeVHighETFixThreshold_v11;
   Int_t           elCand1_isVeryLooseLLH_Smooth_v11;
   Int_t           elCand1_isLooseLLH_Smooth_v11;
   Int_t           elCand1_isLooseAndBLayerLLH_Smooth_v11;
   Int_t           elCand1_isMediumLLH_Smooth_v11;
   Int_t           elCand1_isTightLLH_Smooth_v11;
   Int_t           elCand1_TightLLH_80GeVHighETFixThreshold_Smooth_v11;
   Int_t           elCand1_isTightLLHCalo_v11;
   Int_t           elCand1_isMediumLLHCalo_v11;
   Int_t           elCand1_isLooseLLHCalo_v11;
   Int_t           elCand1_isVeryLooseLLHCalo_v11;
   Int_t           elCand1_isTightLLH_v11;
   Int_t           elCand1_isMediumLLH_v11;
   Int_t           elCand1_isLooseLLH_v11;
   Int_t           elCand1_isVeryLooseLLH_v11;
   Int_t           elCand1_isTightLLHCalo_Smooth_v11;
   Int_t           elCand1_isMediumLLHCalo_Smooth_v11;
   Int_t           elCand1_isLooseLLHCalo_Smooth_v11;
   Int_t           elCand1_isVeryLooseLLHCalo_Smooth_v11;
   Int_t           elCand1_isEMLoose2015;
   Int_t           elCand1_isEMMedium2015;
   Int_t           elCand1_isEMTight2015;
   Int_t           elCand1_isolTight;
   Int_t           elCand1_isolLoose;
   Int_t           elCand1_isolLooseTrackOnly;
   Int_t           elCand1_isolGradient;
   Int_t           elCand1_isolGradientLoose;
   Int_t           elCand1_isolFixedCutTightTrackOnly;
   Int_t           elCand1_FixedCutTight;
   Int_t           elCand1_FixedCutLoose;
   Int_t           elCand1_isTriggerMatched;
   Float_t         elCand1_cl_E;
   Float_t         elCand1_eta;
   Float_t         elCand1_cl_eta;
   Float_t         elCand1_etas2;
   Float_t         elCand1_phi;
   Float_t         elCand1_et;
   Float_t         elCand1_pt;
   Float_t         elCand1_charge;
   Float_t         elCand1_etcone20;
   Float_t         elCand1_etcone30;
   Float_t         elCand1_etcone40;
   Float_t         elCand1_ptcone20;
   Float_t         elCand1_ptcone30;
   Float_t         elCand1_ptcone40;
   Float_t         elCand1_ptvarcone20;
   Float_t         elCand1_ptvarcone30;
   Float_t         elCand1_ptvarcone40;
   Float_t         elCand1_topoetcone20;
   Float_t         elCand1_topoetcone30;
   Float_t         elCand1_topoetcone40;
   Float_t         elCand1_E233;
   Float_t         elCand1_E237;
   Float_t         elCand1_E277;
   Float_t         elCand1_Ethad;
   Float_t         elCand1_Ethad1;
   Float_t         elCand1_f1;
   Float_t         elCand1_f3;
   Float_t         elCand1_weta2;
   Float_t         elCand1_wstot;
   Float_t         elCand1_Emins1;
   Float_t         elCand1_emaxs1;
   Float_t         elCand1_Emax2;
   Float_t         elCand1_fracs1;
   Float_t         elCand1_reta;
   Float_t         elCand1_rhad0;
   Float_t         elCand1_rhad1;
   Float_t         elCand1_rhad;
   Float_t         elCand1_rphi;
   Float_t         elCand1_eratio;
   Float_t         elCand1_deltaeta1;
   Float_t         elCand1_deltaeta2;
   Float_t         elCand1_deltaphi1;
   Float_t         elCand1_deltaphi2;
   Float_t         elCand1_deltaphiRescaled;
   Float_t         elCand1_deltaphiFromLM;
   Float_t         elCand1_trackd0pvunbiased;
   Float_t         elCand1_tracksigd0pvunbiased;
   Float_t         elCand1_d0significance;
   Float_t         elCand1_trackz0pvunbiased;
   Float_t         elCand1_EOverP;
   Float_t         elCand1_z0sinTheta;
   Int_t           elCand1_passd0z0;
   Float_t         elCand1_z0significance;
   Float_t         elCand1_qoverp;
   Float_t         elCand1_qoverpsignificance;
   Float_t         elCand1_chi2oftrackmatch;
   Float_t         elCand1_ndftrackmatch;
   Int_t           elCand1_numberOfInnermostPixelLayerHits;
   Int_t           elCand1_numberOfInnermostPixelLayerOutliers;
   Int_t           elCand1_expectInnermostPixelLayerHit;
   Int_t           elCand1_numberOfNextToInnermostPixelLayerHits;
   Int_t           elCand1_numberOfNextToInnermostPixelLayerOutliers;
   Int_t           elCand1_expectNextToInnermostPixelLayerHit;
   Int_t           elCand1_nBLHits;
   Int_t           elCand1_nBLayerOutliers;
   Int_t           elCand1_expectHitInBLayer;
   Int_t           elCand1_nPixHits;
   Int_t           elCand1_nPixelOutliers;
   Int_t           elCand1_nPixelDeadSensors;
   Int_t           elCand1_nSCTHits;
   Int_t           elCand1_nSCTOutliers;
   Int_t           elCand1_nSCTDeadSensors;
   Int_t           elCand1_nTRTHits;
   Int_t           elCand1_nTRTOutliers;
   Int_t           elCand1_nTRTHighTHits;
   Int_t           elCand1_nTRTHighTOutliers;
   Int_t           elCand1_numberOfTRTDeadStraws;
   Int_t           elCand1_numberOfTRTXenonHits;
   Float_t         elCand1_TRTHighTOutliersRatio;
   Float_t         elCand1_eProbabilityHT;
   Float_t         elCand1_DeltaPOverP;
   Float_t         elCand1_pTErr;
   Float_t         elCand1_deltaCurvOverErrCurv;
   Float_t         elCand1_deltaDeltaPhiFirstAndLM;
   Int_t           elCand1_AmbiguityBit;
   Int_t           elCand1_nBetterMatches;
   Int_t           elCand1_type;
   Int_t           elCand1_origin;
   Int_t           elCand1_originbkg;
   Int_t           elCand1_typebkg;
   Int_t           elCand1_isTruthElectronFromZ;
   Int_t           elCand1_TruthParticlePdgId;
   Int_t           elCand1_firstEgMotherPdgId;
   Int_t           elCand1_TruthParticleBarcode;
   Int_t           elCand1_firstEgMotherBarcode;
   Int_t           elCand1_MotherPdgId;
   Int_t           elCand1_MotherBarcode;
   Int_t           elCand1_FirstEgMotherTyp;
   Int_t           elCand1_FirstEgMotherOrigin;
   Int_t           elCand1_dRPdgId;
   Float_t         elCand1_dR;
   std::vector<float>   *elCand1_ringer_rings;
   Int_t           elCand2_trig_EF_VeryLooseLLH_z0offlineMatch_Smooth_Probe;
   Int_t           elCand2_passTrackQuality;
   Int_t           elCand2_isEMLoose;
   Int_t           elCand2_isEMMedium;
   Int_t           elCand2_isEMTight;
   Int_t           elCand2_isTightPP;
   Int_t           elCand2_isMediumPP;
   Int_t           elCand2_isLoosePP;
   Int_t           elCand2_isVeryLooseLL;
   Int_t           elCand2_isLooseLL;
   Int_t           elCand2_isMediumLL;
   Int_t           elCand2_isTightLL;
   Int_t           elCand2_isVeryTightLL;
   Float_t         elCand2_CFT;
   Int_t           elCand2_passCFT;
   Int_t           elCand2_isVeryLooseLL2015;
   Int_t           elCand2_isLooseLL2015;
   Int_t           elCand2_isMediumLL2015;
   Int_t           elCand2_isTightLL2015;
   Int_t           elCand2_isVeryLooseLL2015_CutD0DphiDeta;
   Int_t           elCand2_isLooseLL2015_CutD0DphiDeta;
   Int_t           elCand2_isMediumLL2015_CutD0DphiDeta;
   Int_t           elCand2_isTightLL2015_CutD0DphiDeta;
   Int_t           elCand2_isLoose2015;
   Int_t           elCand2_isMedium2015;
   Int_t           elCand2_isTight2015;
   Int_t           elCand2_isVeryLooseLL2015_v8;
   Int_t           elCand2_isLooseLL2015_v8;
   Int_t           elCand2_isLooseAndBLayerLL2015_v8;
   Int_t           elCand2_isMediumLL2015_v8;
   Int_t           elCand2_isTightLL2015_v8;
   Int_t           elCand2_isTightLL2015_v10;
   Int_t           elCand2_isVeryLooseLL2016_v11;
   Int_t           elCand2_isLooseLL2016_v11;
   Int_t           elCand2_isLooseAndBLayerLL2016_v11;
   Int_t           elCand2_isMediumLL2016_v11;
   Int_t           elCand2_isTightLL2016_v11;
   Int_t           elCand2_TightLLH_80GeVHighETFixThreshold_v11;
   Int_t           elCand2_isVeryLooseLLH_Smooth_v11;
   Int_t           elCand2_isLooseLLH_Smooth_v11;
   Int_t           elCand2_isLooseAndBLayerLLH_Smooth_v11;
   Int_t           elCand2_isMediumLLH_Smooth_v11;
   Int_t           elCand2_isTightLLH_Smooth_v11;
   Int_t           elCand2_TightLLH_80GeVHighETFixThreshold_Smooth_v11;
   Int_t           elCand2_isEMLoose2015;
   Int_t           elCand2_isEMMedium2015;
   Int_t           elCand2_isEMTight2015;
   Int_t           elCand2_isolTight;
   Int_t           elCand2_isolLoose;
   Int_t           elCand2_isolLooseTrackOnly;
   Int_t           elCand2_isolGradient;
   Int_t           elCand2_isolGradientLoose;
   Int_t           elCand2_isolFixedCutTightTrackOnly;
   Int_t           elCand2_FixedCutTight;
   Int_t           elCand2_FixedCutLoose;
   Int_t           elCand2_isTriggerMatched;
   Int_t           elCand2_passDeltaE;
   Int_t           elCand2_passFSide;
   Int_t           elCand2_passWs3;
   Int_t           elCand2_passEratio;
   Float_t         elCand2_cl_E;
   Float_t         elCand2_eta;
   Float_t         elCand2_cl_eta;
   Float_t         elCand2_etas2;
   Float_t         elCand2_phi;
   Float_t         elCand2_et;
   Float_t         elCand2_pt;
   Float_t         elCand2_charge;
   Float_t         elCand2_etcone20;
   Float_t         elCand2_etcone30;
   Float_t         elCand2_etcone40;
   Float_t         elCand2_ptcone20;
   Float_t         elCand2_ptcone30;
   Float_t         elCand2_ptcone40;
   Float_t         elCand2_ptvarcone20;
   Float_t         elCand2_ptvarcone30;
   Float_t         elCand2_ptvarcone40;
   Float_t         elCand2_topoetcone20;
   Float_t         elCand2_topoetcone30;
   Float_t         elCand2_topoetcone40;
   Float_t         elCand2_E233;
   Float_t         elCand2_E237;
   Float_t         elCand2_E277;
   Float_t         elCand2_Ethad;
   Float_t         elCand2_Ethad1;
   Float_t         elCand2_f1;
   Float_t         elCand2_f3;
   Float_t         elCand2_weta2;
   Float_t         elCand2_wstot;
   Float_t         elCand2_Emins1;
   Float_t         elCand2_emaxs1;
   Float_t         elCand2_Emax2;
   Float_t         elCand2_fracs1;
   Float_t         elCand2_reta;
   Float_t         elCand2_rhad0;
   Float_t         elCand2_rhad1;
   Float_t         elCand2_rhad;
   Float_t         elCand2_rphi;
   Float_t         elCand2_eratio;
   Float_t         elCand2_deltaeta1;
   Float_t         elCand2_deltaeta2;
   Float_t         elCand2_deltaphi1;
   Float_t         elCand2_deltaphi2;
   Float_t         elCand2_deltaphiRescaled;
   Float_t         elCand2_deltaphiFromLM;
   Float_t         elCand2_trackd0pvunbiased;
   Float_t         elCand2_tracksigd0pvunbiased;
   Float_t         elCand2_d0significance;
   Float_t         elCand2_trackz0pvunbiased;
   Float_t         elCand2_EOverP;
   Float_t         elCand2_z0sinTheta;
   Int_t           elCand2_passd0z0;
   Float_t         elCand2_z0significance;
   Float_t         elCand2_qoverp;
   Float_t         elCand2_qoverpsignificance;
   Float_t         elCand2_chi2oftrackmatch;
   Float_t         elCand2_ndftrackmatch;
   Int_t           elCand2_numberOfInnermostPixelLayerHits;
   Int_t           elCand2_numberOfInnermostPixelLayerOutliers;
   Int_t           elCand2_expectInnermostPixelLayerHit;
   Int_t           elCand2_numberOfNextToInnermostPixelLayerHits;
   Int_t           elCand2_numberOfNextToInnermostPixelLayerOutliers;
   Int_t           elCand2_expectNextToInnermostPixelLayerHit;
   Int_t           elCand2_nBLHits;
   Int_t           elCand2_nBLayerOutliers;
   Int_t           elCand2_expectHitInBLayer;
   Int_t           elCand2_nPixHits;
   Int_t           elCand2_nPixelOutliers;
   Int_t           elCand2_nPixelDeadSensors;
   Int_t           elCand2_nSCTHits;
   Int_t           elCand2_nSCTOutliers;
   Int_t           elCand2_nSCTDeadSensors;
   Int_t           elCand2_nTRTHits;
   Int_t           elCand2_nTRTOutliers;
   Int_t           elCand2_nTRTHighTHits;
   Int_t           elCand2_nTRTHighTOutliers;
   Int_t           elCand2_numberOfTRTDeadStraws;
   Int_t           elCand2_numberOfTRTXenonHits;
   Float_t         elCand2_TRTHighTOutliersRatio;
   Float_t         elCand2_eProbabilityHT;
   Float_t         elCand2_DeltaPOverP;
   Float_t         elCand2_pTErr;
   Float_t         elCand2_deltaCurvOverErrCurv;
   Float_t         elCand2_deltaDeltaPhiFirstAndLM;
   Int_t           elCand2_AmbiguityBit;
   Int_t           elCand2_nBetterMatches;
   Int_t           elCand2_type;
   Int_t           elCand2_origin;
   Int_t           elCand2_originbkg;
   Int_t           elCand2_typebkg;
   Int_t           elCand2_isTruthElectronFromZ;
   Int_t           elCand2_TruthParticlePdgId;
   Int_t           elCand2_firstEgMotherPdgId;
   Int_t           elCand2_TruthParticleBarcode;
   Int_t           elCand2_firstEgMotherBarcode;
   Int_t           elCand2_MotherPdgId;
   Int_t           elCand2_MotherBarcode;
   Int_t           elCand2_FirstEgMotherTyp;
   Int_t           elCand2_FirstEgMotherOrigin;
   Int_t           elCand2_dRPdgId;
   Float_t         elCand2_dR;
   std::vector<float>   *elCand2_ringer_rings;
   Int_t           elCand2_isTightLLHCaloMC14Truth;
   Int_t           elCand2_isMediumLLHCaloMC14Truth;
   Int_t           elCand2_isLooseLLHCaloMC14Truth;
   Int_t           elCand2_isVeryLooseLLHCaloMC14Truth;
   Int_t           elCand2_isTightLLHCaloMC14;
   Int_t           elCand2_isMediumLLHCaloMC14;
   Int_t           elCand2_isLooseLLHCaloMC14;
   Int_t           elCand2_isVeryLooseLLHCaloMC14;
   Int_t           elCand2_isTightLLHMC15Calo_v8;
   Int_t           elCand2_isMediumLLHMC15Calo_v8;
   Int_t           elCand2_isLooseLLHMC15Calo_v8;
   Int_t           elCand2_isVeryLooseLLHMC15Calo_v8;
   Int_t           elCand2_isTightLLHCalo_v11;
   Int_t           elCand2_isMediumLLHCalo_v11;
   Int_t           elCand2_isLooseLLHCalo_v11;
   Int_t           elCand2_isVeryLooseLLHCalo_v11;
   Int_t           elCand2_isTightLLHMC15_v8;
   Int_t           elCand2_isMediumLLHMC15_v8;
   Int_t           elCand2_isLooseLLHMC15_v8;
   Int_t           elCand2_isLooseAndBLayerLLHMC15_v8;
   Int_t           elCand2_isVeryLooseLLHMC15_v8;
   Int_t           elCand2_isTightLLH_v11;
   Int_t           elCand2_isMediumLLH_v11;
   Int_t           elCand2_isLooseLLH_v11;
   Int_t           elCand2_isVeryLooseLLH_v11;
   Int_t           elCand2_isTightLLHCalo_Smooth_v11;
   Int_t           elCand2_isMediumLLHCalo_Smooth_v11;
   Int_t           elCand2_isLooseLLHCalo_Smooth_v11;
   Int_t           elCand2_isVeryLooseLLHCalo_Smooth_v11;

   Int_t           fcCand1_match;
   Float_t         fcCand1_cl_E;
   Float_t         fcCand1_eta;
   Float_t         fcCand1_phi;
   Float_t         fcCand1_et;
   Float_t         fcCand1_E237;
   Float_t         fcCand1_E277;
   Float_t         fcCand1_Ethad;
   Float_t         fcCand1_Ethad1;
   Float_t         fcCand1_f1;
   Float_t         fcCand1_f3;
   Float_t         fcCand1_weta2;
   Float_t         fcCand1_wstot;
   Float_t         fcCand1_emaxs1;
   Float_t         fcCand1_Emax2;
   Float_t         fcCand1_fracs1;
   Float_t         fcCand1_reta;
   Float_t         fcCand1_rhad0;
   Float_t         fcCand1_rhad1;
   Float_t         fcCand1_rhad;
   Float_t         fcCand1_eratio;
   Int_t           fcCand1_etaBin;
   Int_t           fcCand1_eTBin;
   Int_t           fcCand1_ringerMatch;
   std::vector<float>   *fcCand1_ringer_rings;
   Int_t           fcCand1_ringerEtaBin;
   Int_t           fcCand1_ringerETBin;
   Int_t           fcCand2_match;
   Float_t         fcCand2_cl_E;
   Float_t         fcCand2_eta;
   Float_t         fcCand2_phi;
   Float_t         fcCand2_et;
   Float_t         fcCand2_E237;
   Float_t         fcCand2_E277;
   Float_t         fcCand2_Ethad;
   Float_t         fcCand2_Ethad1;
   Float_t         fcCand2_f1;
   Float_t         fcCand2_f3;
   Float_t         fcCand2_weta2;
   Float_t         fcCand2_wstot;
   Float_t         fcCand2_emaxs1;
   Float_t         fcCand2_Emax2;
   Float_t         fcCand2_fracs1;
   Float_t         fcCand2_reta;
   Float_t         fcCand2_rhad0;
   Float_t         fcCand2_rhad1;
   Float_t         fcCand2_rhad;
   Float_t         fcCand2_eratio;
   Int_t           fcCand2_etaBin;
   Int_t           fcCand2_eTBin;
   Int_t           fcCand2_ringerMatch;
   std::vector<float>   *fcCand2_ringer_rings;
   Int_t           fcCand2_ringerEtaBin;
   Int_t           fcCand2_ringerETBin;

};

#endif // #ifdef TUNINGTOOLS_SKIMMEDNTUPLE_H
