cmake_minimum_required(VERSION 3.24)
project(gwphys_cuda LANGUAGES CXX CUDA)

# Python e pybind11 do seu venv
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 REQUIRED)

# Arquivos
set(SRC
    gwphys_cuda.cu
)

# Módulo Python
pybind11_add_module(gwphys_cuda MODULE ${SRC})

# Padrões e flags
target_compile_features(gwphys_cuda PRIVATE cxx_std_17)
set_target_properties(gwphys_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX "" 
)

# Arquitetura da sua GPU
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # RTX 3060
endif()

# Includes
target_include_directories(gwphys_cuda PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Flags CUDA - extended lambdas, fast math, line info
target_compile_options(gwphys_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math -lineinfo>
)

# Se você usar cuBLAS ou cuFFT futuramente, ligue aqui
# find_package(CUDAToolkit REQUIRED)
# target_link_libraries(gwphys_cuda PRIVATE CUDA::cudart CUDA::cublas CUDA::cufft)

# No Linux, exporta com sufixo .so no lugar de .cpython... se quiser:
# set_target_properties(gwphys_cuda PROPERTIES SUFFIX ".so")

# Otimização LTO quando disponível
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok OUTPUT ipo_msg)
if(ipo_ok)
  set_property(TARGET gwphys_cuda PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()