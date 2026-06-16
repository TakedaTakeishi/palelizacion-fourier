CC = mpicc
CFLAGS = -Wall -Wextra -O3
LDFLAGS = -lm

TARGET = fourier_mpi
SRC = fourier_mpi.c
NP = 4

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

run: $(TARGET)
	mpirun --mca btl ^tcp -np $(NP) ./$(TARGET)

run_mpi: $(TARGET)
	mpirun --mca btl ^tcp -np $(NP) ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run run_mpi clean