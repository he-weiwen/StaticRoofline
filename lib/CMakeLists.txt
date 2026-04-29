# Expose the lib/ directory so PTX/*.cpp can include "OpClassifier.h"
# and Classifier.h can include "PTX/Parser.h" with stable paths.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_llvm_library(NVPTXArithIntensity MODULE
  NVPTXArithIntensityPass.cpp
  OpClassifier.cpp
  PTX/Tokenizer.cpp
  PTX/Parser.cpp
  PTX/Classifier.cpp
  PLUGIN_TOOL llc)
