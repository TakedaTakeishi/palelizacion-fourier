CC = mpicc
CFLAGS = -Wall -Wextra -O3
LDFLAGS = 

TARGET = main
SRC = main.c
MPI_TARGET = fourier_mpi
MPI_SRC = fourier_mpi.c

all: $(TARGET) $(MPI_TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

$(MPI_TARGET): $(MPI_SRC)
	$(CC) $(CFLAGS) -o $(MPI_TARGET) $(MPI_SRC) $(LDFLAGS)

run: $(TARGET)
	mpirun --mca btl ^tcp -np 8 ./$(TARGET)

run_mpi: $(MPI_TARGET)
	mpirun --mca btl ^tcp -np 4 ./$(MPI_TARGET)

clean:
	rm -f $(TARGET) $(MPI_TARGET)

.PHONY: all run run_mpi clean
