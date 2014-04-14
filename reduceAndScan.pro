#-------------------------------------------------
#
# Project created by QtCreator 2014-03-17T12:00:53
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = reduceAndScan
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp

HEADERS  += \
    modernGPU.cuh

OTHER_FILES += ./modernGPU.cu

# CUDA settings
CUDA_SOURCES += ./modernGPU.cu

CUDA_SDK = "/usr/local/cuda-5.5/"   # Path to cuda SDK install
CUDA_DIR = "/usr/local/cuda-5.5"            # Path to cuda toolkit install
SYSTEM_NAME = unix         # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_21           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math

DEFINES += CUDA_ARCH=\\\"$${CUDA_ARCH}\\\"

# include paths
INCLUDEPATH += $$CUDA_DIR/include

# library directories
QMAKE_LIBDIR += $$CUDA_DIR/lib64

CUDA_OBJECTS_DIR = ./

# Add the necessary libraries
CUDA_LIBS = -lcuda -lcudart
# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
LIBS += $$CUDA_LIBS

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    # --machine $$SYSTEM_TYPE
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    #--machine $$SYSTEM_TYPE
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
