# Expose the lib/ directory so PTX/*.cpp can include "OpClassifier.h"
# and Classifier.h can include "PTX/Parser.h" with stable paths.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Tell LLVM's add_llvm_library that these .cpp files are intentionally
# not direct sources of the .so target — they're consumed via
# $<TARGET_OBJECTS:ptxai_ptx>. Without this, LLVMProcessSources.cmake
# errors out on "erroneous configuration for source file".
set(LLVM_OPTIONAL_SOURCES
  PTX/Tokenizer.cpp
  PTX/Parser.cpp
  PTX/Classifier.cpp
  OpClassifier.cpp
  Measurement.cpp
  Stats.cpp)

# OBJECT library: same source files compiled into both the plugin (.so)
# and the standalone unit-test binary (test/ptx_unit_tests). Object form
# avoids re-compiling and keeps the link surface narrow.
add_library(ptxai_ptx OBJECT
  PTX/Tokenizer.cpp
  PTX/Parser.cpp
  PTX/Classifier.cpp
  OpClassifier.cpp
  Measurement.cpp
  Stats.cpp)
set_property(TARGET ptxai_ptx PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(ptxai_ptx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_llvm_library(NVPTXArithIntensity MODULE
  NVPTXArithIntensityPass.cpp
  $<TARGET_OBJECTS:ptxai_ptx>
  PLUGIN_TOOL llc)
