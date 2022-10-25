cmake_minimum_required(VERSION 3.16)
project(test_keras_model)

include(FetchContent)
FetchContent_Declare(
    functional_plus
    GIT_REPOSITORY https://github.com/Dobiasd/FunctionalPlus.git
    GIT_TAG master
    )

  FetchContent_Declare(
    frugally_deep
    GIT_REPOSITORY https://github.com/Dobiasd/frugally-deep.git
    GIT_TAG master
)

FetchContent_MakeAvailable(functional_plus frugally_deep)

find_package(Eigen3)
find_package(YARP)

add_executable(test_keras_model main.cpp)
target_link_libraries(test_keras_model frugally-deep::fdeep Eigen3::Eigen ${YARP_LIBRARIES})
