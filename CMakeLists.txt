cmake_minimum_required(VERSION 3.24)
project(HNSW_PCA)

set(CMAKE_CXX_STANDARD 17)

set(DCMAKE_BUILD_TYPE=Release)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3")


find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

# generate .so file

include_directories(inc)

add_library(hnswc
        SHARED
        src/utils.cpp
        src/pca.cpp
        src/hnsw.cpp
        )

add_executable(hnsw_test src/test.cpp)
target_link_libraries(hnsw_test hnswc)
