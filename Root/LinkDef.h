#include "TuningTools/MuonPhysVal.h"
#include "TuningTools/RingerPhysVal.h"
#include "TuningTools/RingerPhysVal_v2.h"
#include "TuningTools/SkimmedNtuple.h"
#include <vector>
#include <utility>

#if defined(__CLING__) || defined(__CINT__)
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link C++ nestedclass;

// Create dictionaries for the used vector types:
//#pragma link C++ class std::vector<float>+;
//#pragma link C++ class std::vector< std::vector<float> >+;
#pragma link C++ class std::vector<int8_t>+;
#pragma link C++ class std::vector<std::pair<std::string,int>+;

// And for the event model class:
#pragma link C++ class RingerPhysVal+;
#pragma link C++ class RingerPhysVal_v2+;
#pragma link C++ class MuonPhysVal+;
#pragma link C++ class SkimmedNtuple+;

#endif
