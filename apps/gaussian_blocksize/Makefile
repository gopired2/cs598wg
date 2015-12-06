
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/opt/cuda/cuda-6.5/include
LD_FLAGS    = -lcudart -L/opt/cuda/cuda-6.5/lib64
EXE	        = gaussian
OBJ	        = gaussian.o

default: $(EXE)

gaussian.o: gaussian.cu
	$(NVCC) -c -o $@ gaussian.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
