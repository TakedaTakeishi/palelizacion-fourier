# Makefile para el Proyecto Final: Serie de Fourier Paralela
# Compilador a usar
CC = gcc
# Opciones de compilación (advertencias + librería matemática)
CFLAGS = -Wall
LDFLAGS = -lm

# Nombre del ejecutable
EXEC = fourier

# Regla por defecto
all: $(EXEC)

# Compilar fourier.c
$(EXEC): fourier.c
	$(CC) $(CFLAGS) -o $(EXEC) fourier.c $(LDFLAGS)

# Ejecutar el programa
run: $(EXEC)
	./$(EXEC)

# Limpiar archivos generados
clean:
	rm -f $(EXEC) hoja1.csv hoja2.csv

.PHONY: all run clean
