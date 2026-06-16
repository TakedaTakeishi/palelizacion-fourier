CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lm
EXEC = fourier_seq

all: $(EXEC)

$(EXEC): fourier_seq.c
	$(CC) $(CFLAGS) -o $(EXEC) fourier_seq.c $(LDFLAGS)

run: $(EXEC)
	./$(EXEC)

clean:
	rm -f $(EXEC) hoja1.csv hoja2.csv

.PHONY: all run clean
