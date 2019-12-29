CC=/usr/local/cuda/bin/nvcc
CFLAGS=-I/usr/local/cuda-10.1/include/ -I/usr/local/cuda-10.1/samples/common/inc/ -L/usr/local/cuda-10.1/lib64 -L/usr/local/pgplot/ -lcufft
DEPS = filterbank_cuda.hpp
ASM = nasm -f elf64
AFLAGS =


CPP = g++ -g -Wall
ALL_CFLAGS = -ccbin g++ -I/usr/local/cuda-10.1/include/ -I/usr/local/cuda-10.1/samples/common/inc/  -m64 \
	-gencode arch=compute_35,code=\"sm_35,compute_35\" -lcufft -std=c++11
LD_FLAGS = -lm -lpthread -lcpgplot -lcufft -std=c++11 -fPIC -march=native -ffast-math -funroll-loops
OBJ = gsb2dat.o gpu_kernels.o

FILES = \
    
%.o: %.c
	$(CC) $(ALL_CFLAGS) -o $@ -c $<

%.o: %.cu
	$(CC) $(ALL_CFLAGS) -o $@ -c $<

%.o: %.cpp
	$(CPP) $(CFLAGS) -o $@ -c $< $(LD_FLAGS)

%.o: %.asm
	$(ASM) $(AFLAGS) -o $@ $<

gsb2dat: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f -r ./*.o

