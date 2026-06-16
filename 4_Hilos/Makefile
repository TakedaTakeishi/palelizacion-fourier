CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm -lpthread
EXEC = hilos_fourier

all: $(EXEC)

$(EXEC): hilos_fourier.c
	$(CC) $(CFLAGS) -o $(EXEC) hilos_fourier.c $(LDFLAGS)

run: $(EXEC)
	./$(EXEC)

clean:
	rm -f $(EXEC) hoja1.csv hoja2.csv

.PHONY: all run clean
