################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../playground/print_kernel.o 

CPP_SRCS += \
../playground/print_threads.cpp 

CU_SRCS += \
../playground/print_kernel.cu 

CU_DEPS += \
./playground/print_kernel.d 

OBJS += \
./playground/print_kernel.o \
./playground/print_threads.o 

CPP_DEPS += \
./playground/print_threads.d 


# Each subdirectory must supply rules for building sources it contributes
playground/%.o: ../playground/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "playground" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_20  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

playground/%.o: ../playground/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 -gencode arch=compute_20,code=sm_20  -odir "playground" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


