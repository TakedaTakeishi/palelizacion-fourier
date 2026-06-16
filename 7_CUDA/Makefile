# Makefile para el Proyecto Final: Serie de Fourier Paralela (CUDA)
# Compiladores a usar
NVCC = nvcc

# Opciones de compilación (advertencias + librería matemática)
NVCCFLAGS = -O3
LDFLAGS = -lm

# Nombre del ejecutable
EXEC = fourier

# Regla por defecto (compilar CUDA)
all: $(EXEC)

# Compilar fourier.cu con nvcc
$(EXEC): fourier.cu
	$(NVCC) $(NVCCFLAGS) -o $(EXEC) fourier.cu $(LDFLAGS)

# Ejecutar el programa de CUDA
run: $(EXEC)
	./$(EXEC)

# Limpiar archivos generados
clean:
	rm -f $(EXEC) "resultados_f(x).csv" resultados_fourier.csv

.PHONY: all run clean
