set(CPP_SOURCE_FILES 
    data.cpp 
    knn_parser.cpp
    caster/caster_ab.cpp
    caster/caster_nesterov.cpp
    caster/caster_adadelta_async.cpp
    caster/caster_adadelta_sync.cpp
)
set(CUDA_SOURCE_FILES
    caster/caster_cuda.cu
    caster/caster_cuda_ab.cu
    caster/caster_cuda_nesterov.cu
    caster/caster_cuda_adadelta.cu
    caster/caster_cuda_adam.cu
    main.cpp
)

CUDA_ADD_EXECUTABLE(${EXECUTABLE_NAME} ${CPP_SOURCE_FILES} ${CUDA_SOURCE_FILES})

# Link shared libraries and compiled CUDA libraries
TARGET_LINK_LIBRARIES (${EXECUTABLE_NAME} ${CUDA_LIBRARIES})
