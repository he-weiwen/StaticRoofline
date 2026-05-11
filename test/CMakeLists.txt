set(LLC_EXE ${LLVM_TOOLS_BINARY_DIR}/llc)
set(CLANGXX_EXE ${LLVM_TOOLS_BINARY_DIR}/clang++)
set(FILECHECK_EXE ${LLVM_TOOLS_BINARY_DIR}/FileCheck)

# Default arch for tests. sm_70 is enough for the IR-level tests; sm_80 is
# the floor for bf16x2 / Ampere-and-later opcodes that the CUDA tests use.
set(PTXAI_TEST_GPU_ARCH "sm_80" CACHE STRING "GPU arch for ptxai test compilation")
# CUDA toolkit path for the .cu test pipeline. Overridable on the command line.
set(PTXAI_CUDA_PATH "/usr/local/cuda-13.0" CACHE PATH "CUDA toolkit path for .cu test compilation")

# Define a single ptx-ai test from a .ll file containing FileCheck patterns.
# Lowers <name>.ll to MIR via llc, runs the plugin, pipes output through
# FileCheck against the same .ll.
function(add_ptxai_test name)
  set(LL ${CMAKE_CURRENT_SOURCE_DIR}/${name}.ll)
  set(MIR ${CMAKE_CURRENT_BINARY_DIR}/${name}.mir)

  add_custom_command(
    OUTPUT ${MIR}
    COMMAND ${LLC_EXE} -march=nvptx64 -mcpu=${PTXAI_TEST_GPU_ARCH} -O2
            ${LL} -stop-before=nvptx-asm-printer -o ${MIR}
    DEPENDS ${LL}
    COMMENT "Lowering ${name}.ll to MIR"
    VERBATIM)

  add_custom_target(test-${name}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run-test.sh
            ${LLC_EXE}
            $<TARGET_FILE:NVPTXArithIntensity>
            ${MIR}
            ${LL}
            ${FILECHECK_EXE}
    DEPENDS NVPTXArithIntensity ${MIR}
    COMMENT "Checking ${name}"
    VERBATIM)
endfunction()

# Same idea but starting from a .cu file: clang compiles to .ll, then the
# usual .ll → .mir → run-pass → FileCheck pipeline. Uses the .cu itself as
# the FileCheck input (CHECK lines live in C++ comments).
function(add_ptxai_cuda_test name)
  set(CU ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cu)
  set(LL ${CMAKE_CURRENT_BINARY_DIR}/${name}.ll)
  set(MIR ${CMAKE_CURRENT_BINARY_DIR}/${name}.mir)

  add_custom_command(
    OUTPUT ${LL}
    COMMAND ${CLANGXX_EXE}
            --cuda-device-only --cuda-gpu-arch=${PTXAI_TEST_GPU_ARCH}
            --cuda-path=${PTXAI_CUDA_PATH}
            -emit-llvm -S -O2 -nocudalib -ffp-contract=fast
            ${CU} -o ${LL}
    DEPENDS ${CU}
    COMMENT "Compiling ${name}.cu to LLVM IR"
    VERBATIM)

  add_custom_command(
    OUTPUT ${MIR}
    COMMAND ${LLC_EXE} -march=nvptx64 -mcpu=${PTXAI_TEST_GPU_ARCH} -O2
            ${LL} -stop-before=nvptx-asm-printer -o ${MIR}
    DEPENDS ${LL}
    COMMENT "Lowering ${name}.ll to MIR"
    VERBATIM)

  add_custom_target(test-${name}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run-test.sh
            ${LLC_EXE}
            $<TARGET_FILE:NVPTXArithIntensity>
            ${MIR}
            ${CU}
            ${FILECHECK_EXE}
    DEPENDS NVPTXArithIntensity ${MIR}
    COMMENT "Checking ${name}"
    VERBATIM)
endfunction()

add_ptxai_test(smoke)
add_ptxai_test(fsub)
add_ptxai_test(add_f64)
add_ptxai_test(local)
add_ptxai_test(fma)
add_ptxai_cuda_test(fma_v2half)
add_ptxai_cuda_test(matmul_loops)
add_ptxai_cuda_test(matmul_tiled_loops)
add_ptxai_cuda_test(inline_asm_hfma2)
add_ptxai_cuda_test(ldg_restrict)

# Standalone unit-test binary for the lib/PTX/ tokenizer/parser/classifier
# pipeline. Linked against the same object files as the plugin via
# $<TARGET_OBJECTS:ptxai_ptx>, so it tests exactly what runs in production.
add_executable(ptx_unit_tests
  ptx_unit_tests.cpp
  $<TARGET_OBJECTS:ptxai_ptx>)
target_include_directories(ptx_unit_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/lib
  ${LLVM_INCLUDE_DIRS})
# Link against LLVM Support for StringRef / SmallVector machinery.
llvm_map_components_to_libnames(_unit_test_libs Support)
target_link_libraries(ptx_unit_tests PRIVATE ${_unit_test_libs})

add_custom_target(test-ptx-unit
  COMMAND $<TARGET_FILE:ptx_unit_tests>
  DEPENDS ptx_unit_tests
  COMMENT "Running PTX tokenizer/parser/classifier unit tests"
  USES_TERMINAL)

add_custom_target(check
  DEPENDS test-smoke test-fsub test-add_f64 test-local test-fma
          test-fma_v2half test-matmul_loops test-matmul_tiled_loops
          test-inline_asm_hfma2 test-ldg_restrict
          test-ptx-unit
  COMMENT "Running all ptx-ai tests")

# Backward-compat alias for the previous target name.
add_custom_target(smoke-test DEPENDS test-smoke)
