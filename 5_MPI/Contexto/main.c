#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/*
Práctica 5: Programación Paralela con MPI
Alumnos:
Bustillos Cruz Jonatan
Delgado Lucero Cristian Isaac
Frem Cortés José Angel
Luna Gonzales Gabriel Alexis
Grupo: 6BV1
Fecha: 16/05/2026
*/

#define N_COLUMNAS 3

int main(int argc, char **argv) {
  int pid, procesos, destino, origen;
  int permiso = 1; // Token de sincronización para salida en orden
  double fila[N_COLUMNAS];
  double resultado;
  MPI_Status estatus;

  // Matriz de datos (a, b, c) para cada nodo esclavo
  double matriz[7][3] = {
      {1, 2, 3},     // Nodo 1
      {4, 5, 6},     // Nodo 2
      {7, 8, 9},     // Nodo 3
      {9, 6, 3},     // Nodo 4
      {-6, -9, -12}, // Nodo 5
      {5, 10, 15},   // Nodo 6
      {7, 14, 21}    // Nodo 7
  };

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &procesos);

  if (procesos != 8) {
    if (pid == 0) {
      printf("Error: Se requieren exactamente 8 procesos.\n");
    }
    MPI_Finalize();
    return 0;
  }

  if (pid == 0) {
    // --- NODO MAESTRO ---
    printf("--------------------------------------------------\n");
    printf("   PRACTICA 5 - MANEJO DE MPI                     \n");
    printf("--------------------------------------------------\n\n");
    fflush(stdout);

    // Mandar los datos a cada uno de los 7 esclavos
    for (destino = 1; destino <= 7; destino++) {
      printf("[Maestro] (Tiempo MPI: %.6f) Enviando fila %d (a=%.2f, b=%.2f, "
             "c=%.2f) al Nodo %d\n",
             MPI_Wtime(), destino, matriz[destino - 1][0],
             matriz[destino - 1][1], matriz[destino - 1][2], destino);
      fflush(stdout);
      MPI_Send(&matriz[destino - 1][0], N_COLUMNAS, MPI_DOUBLE, destino, 100,
               MPI_COMM_WORLD);
    }

    printf("\n[Maestro] Esperando resultados...\n\n");
    fflush(stdout);

    // Recibir los resultados de cada uno de los 7 esclavos en orden
    // sincronizado
    for (origen = 1; origen <= 7; origen++) {
      // 1. Dar permiso al nodo esclavo para que imprima sus mensajes y envíe su
      // resultado
      MPI_Send(&permiso, 1, MPI_INT, origen, 150, MPI_COMM_WORLD);

      // 2. Recibir el resultado del nodo esclavo
      MPI_Recv(&resultado, 1, MPI_DOUBLE, origen, 200, MPI_COMM_WORLD,
               &estatus);
      printf("[Maestro] (Tiempo MPI: %.6f) Resultado recibido del Nodo %d: "
             "%.2f\n\n",
             MPI_Wtime(), origen, resultado);
      fflush(stdout);
      usleep(20000); // Pequeña pausa para asegurar la correcta separación en la
                     // terminal
    }

    printf("--------------------------------------------------\n");
    printf("   TODOS LOS PROCESOS HAN TERMINADO EXITOSAMENTE  \n");
    printf("--------------------------------------------------\n");
    fflush(stdout);

  } else if (pid == 1) {
    // Operación: a + b + c
    MPI_Recv(fila, N_COLUMNAS, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &estatus);
    resultado = fila[0] + fila[1] + fila[2];

    // Esperar el token de permiso del maestro para imprimir y enviar
    MPI_Recv(&permiso, 1, MPI_INT, 0, 150, MPI_COMM_WORLD, &estatus);

    printf("[Nodo 1]  (Tiempo MPI: %.6f) Recibí elementos a=%.2f, b=%.2f, "
           "c=%.2f\n",
           MPI_Wtime(), fila[0], fila[1], fila[2]);
    printf("[Nodo 1]  (Tiempo MPI: %.6f) Enviando resultado de (a + b + c) = "
           "%.2f al Maestro\n",
           MPI_Wtime(), resultado);
    fflush(stdout);
    usleep(20000); // Dar tiempo a que mpirun muestre esto en la terminal antes
                   // de que el maestro reciba

    MPI_Send(&resultado, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);

  } else if (pid == 2) {
    // Operación: a - b + c
    MPI_Recv(fila, N_COLUMNAS, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &estatus);
    resultado = fila[0] - fila[1] + fila[2];

    // Esperar el token de permiso del maestro para imprimir y enviar
    MPI_Recv(&permiso, 1, MPI_INT, 0, 150, MPI_COMM_WORLD, &estatus);

    printf("[Nodo 2]  (Tiempo MPI: %.6f) Recibí elementos a=%.2f, b=%.2f, "
           "c=%.2f\n",
           MPI_Wtime(), fila[0], fila[1], fila[2]);
    printf("[Nodo 2]  (Tiempo MPI: %.6f) Enviando resultado de (a - b + c) = "
           "%.2f al Maestro\n",
           MPI_Wtime(), resultado);
    fflush(stdout);
    usleep(20000);

    MPI_Send(&resultado, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);

  } else if (pid == 3) {
    // Operación: a * b * c
    MPI_Recv(fila, N_COLUMNAS, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &estatus);
    resultado = fila[0] * fila[1] * fila[2];

    // Esperar el token de permiso del maestro para imprimir y enviar
    MPI_Recv(&permiso, 1, MPI_INT, 0, 150, MPI_COMM_WORLD, &estatus);

    printf("[Nodo 3]  (Tiempo MPI: %.6f) Recibí elementos a=%.2f, b=%.2f, "
           "c=%.2f\n",
           MPI_Wtime(), fila[0], fila[1], fila[2]);
    printf("[Nodo 3]  (Tiempo MPI: %.6f) Enviando resultado de (a * b * c) = "
           "%.2f al Maestro\n",
           MPI_Wtime(), resultado);
    fflush(stdout);
    usleep(20000);

    MPI_Send(&resultado, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);

  } else if (pid == 4) {
    // Operación: a * (b + c)
    MPI_Recv(fila, N_COLUMNAS, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &estatus);
    resultado = fila[0] * (fila[1] + fila[2]);

    // Esperar el token de permiso del maestro para imprimir y enviar
    MPI_Recv(&permiso, 1, MPI_INT, 0, 150, MPI_COMM_WORLD, &estatus);

    printf("[Nodo 4]  (Tiempo MPI: %.6f) Recibí elementos a=%.2f, b=%.2f, "
           "c=%.2f\n",
           MPI_Wtime(), fila[0], fila[1], fila[2]);
    printf("[Nodo 4]  (Tiempo MPI: %.6f) Enviando resultado de a * (b + c) = "
           "%.2f al Maestro\n",
           MPI_Wtime(), resultado);
    fflush(stdout);
    usleep(20000);

    MPI_Send(&resultado, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);

  } else if (pid == 5) {
    // Operación: a * b - b * c
    MPI_Recv(fila, N_COLUMNAS, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &estatus);
    resultado = (fila[0] * fila[1]) - (fila[1] * fila[2]);

    // Esperar el token de permiso del maestro para imprimir y enviar
    MPI_Recv(&permiso, 1, MPI_INT, 0, 150, MPI_COMM_WORLD, &estatus);

    printf("[Nodo 5]  (Tiempo MPI: %.6f) Recibí elementos a=%.2f, b=%.2f, "
           "c=%.2f\n",
           MPI_Wtime(), fila[0], fila[1], fila[2]);
    printf("[Nodo 5]  (Tiempo MPI: %.6f) Enviando resultado de (a*b - b*c) = "
           "%.2f al Maestro\n",
           MPI_Wtime(), resultado);
    fflush(stdout);
    usleep(20000);

    MPI_Send(&resultado, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);

  } else if (pid == 6) {
    // Operación: c * (a + b)^2
    MPI_Recv(fila, N_COLUMNAS, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &estatus);
    resultado = fila[2] * (fila[0] + fila[1]) * (fila[0] + fila[1]);

    // Esperar el token de permiso del maestro para imprimir y enviar
    MPI_Recv(&permiso, 1, MPI_INT, 0, 150, MPI_COMM_WORLD, &estatus);

    printf("[Nodo 6]  (Tiempo MPI: %.6f) Recibí elementos a=%.2f, b=%.2f, "
           "c=%.2f\n",
           MPI_Wtime(), fila[0], fila[1], fila[2]);
    printf("[Nodo 6]  (Tiempo MPI: %.6f) Enviando resultado de c * (a + b)^2 = "
           "%.2f al Maestro\n",
           MPI_Wtime(), resultado);
    fflush(stdout);
    usleep(20000);

    MPI_Send(&resultado, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);

  } else if (pid == 7) {
    // Operación: a^2 - b^2 + c^3
    MPI_Recv(fila, N_COLUMNAS, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &estatus);
    resultado = (fila[0] * fila[0]) - (fila[1] * fila[1]) +
                (fila[2] * fila[2] * fila[2]);

    // Esperar el token de permiso del maestro para imprimir y enviar
    MPI_Recv(&permiso, 1, MPI_INT, 0, 150, MPI_COMM_WORLD, &estatus);

    printf("[Nodo 7]  (Tiempo MPI: %.6f) Recibí elementos a=%.2f, b=%.2f, "
           "c=%.2f\n",
           MPI_Wtime(), fila[0], fila[1], fila[2]);
    printf("[Nodo 7]  (Tiempo MPI: %.6f) Enviando resultado de (a^2 - b^2 + "
           "c^3) = %.2f al Maestro\n",
           MPI_Wtime(), resultado);
    fflush(stdout);
    usleep(20000);

    MPI_Send(&resultado, 1, MPI_DOUBLE, 0, 200, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
