cmake_minimum_required(VERSION 3.12)
project(calc_engine_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(OPENMP "Enable OpenMP" ON)
if(OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()

# Otimização padrão
if (MSVC)
  add_compile_options(/O2)
else()
  add_compile_options(-O3 -march=native)
endif()

add_executable(calc_engine
  src/main.cpp
  src/engine.cpp
)
target_include_directories(calc_engine PRIVATE src)
