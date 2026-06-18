#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../common/fourier_core.h"

#define CSV_HOJA1 "hoja1.csv"
#define CSV_HOJA2 "hoja2.csv"

/* Device calculation helper functions */
__device__ double f_device(double x, int func_type) {
    switch (func_type) {
        case FUNC_X4:
            return (x * x * x * x) - 3.0 * x;
        case FUNC_SQUARE:
            return (x > 0) ? 1.0 : (x < 0) ? -1.0 : 0.0;
        case FUNC_SAWTOOTH: {
            double xn = fmod(x + M_PI, 2.0 * M_PI);
            if (xn < 0) xn += 2.0 * M_PI;
            xn -= M_PI;
            if (xn <= -M_PI + 1e-12 || xn >= M_PI - 1e-12) return 0.0;
            return xn / M_PI;
        }
        case FUNC_TRIANGLE: {
            double xn = fmod(x + M_PI, 2.0 * M_PI);
            if (xn < 0) xn += 2.0 * M_PI;
            xn -= M_PI;
            if (xn <= -M_PI + 1e-12 || xn >= M_PI - 1e-12) return 0.0;
            if (xn >= -M_PI/2.0 && xn <= M_PI/2.0) return 2.0 * xn / M_PI;
            if (xn < -M_PI/2.0) return -2.0 * (xn + M_PI) / M_PI;
            return -2.0 * (xn - M_PI) / M_PI;
        }
        default:
            return 0.0;
    }
}

__device__ double termino_fourier_device(long long n, double x, int func_type) {
    switch (func_type) {
        case FUNC_X4: {
            double n2 = (double)n * (double)n;
            double n4 = n2 * n2;
            double signo = (n % 2 == 0) ? 1.0 : -1.0;
            double pi2 = M_PI * M_PI;
            double an = (8.0 * pi2 * n2 - 48.0) * signo / n4;
            double bn = 6.0 * signo / (double)n;
            return an * cos((double)n * x) + bn * sin((double)n * x);
        }
        case FUNC_SQUARE: {
            if (n % 2 == 0) return 0.0;
            return (4.0 / (M_PI * (double)n)) * sin((double)n * x);
        }
        case FUNC_SAWTOOTH: {
            double signo = (n % 2 == 0) ? -1.0 : 1.0;
            return (2.0 * signo / (M_PI * (double)n)) * sin((double)n * x);
        }
        case FUNC_TRIANGLE: {
            if (n % 2 == 0) return 0.0;
            long long k = (n - 1) / 2;
            double signo = (k % 2 == 0) ? 1.0 : -1.0;
            return (8.0 * signo / (M_PI * M_PI * (double)n * (double)n)) * sin((double)n * x);
        }
        default:
            return 0.0;
    }
}

__global__ void calcular_fourier_kernel(DatosFourier *datos) {
  int fila = blockIdx.x;
  if (fila >= NUM_PUNTOS) return;

  double x = datos->x_vals[fila];
  int tid = threadIdx.x;
  int block_dim = blockDim.x;

  double thread_sum = 0.0;

  // Grid-stride loop over terms for this specific point
  for (long long n = tid + 1; n <= datos->num_terminos; n += block_dim) {
      double term = termino_fourier_device(n, x, datos->func_type);
      if (n <= MAX_SAVED_TERMINOS) {
          datos->hoja2_terminos[fila][n - 1] = term;
      }
      thread_sum += term;
  }

  // Shared memory for block reduction
  __shared__ double s_data[256];
  s_data[tid] = thread_sum;
  __syncthreads();

  // Perform block reduction
  for (unsigned int s = block_dim / 2; s > 0; s >>= 1) {
      if (tid < s) {
          s_data[tid] += s_data[tid + s];
      }
      __syncthreads();
  }

  // Thread 0 writes the final result for this point x
  if (tid == 0) {
      datos->hoja1_fx[fila] = f_device(x, datos->func_type);
      datos->hoja2_fx[fila] = f_device(x, datos->func_type);
      datos->hoja2_Fx[fila] = (datos->hoja2_a0 / 2.0) + s_data[0];
  }
}

static void generar_puntos_x(double *x_vals) {
    x_vals[0] = -3.1416;
    for (int i = 1; i <= 62; i++) {
        x_vals[i] = -3.1416 + i * 0.1;
    }
    x_vals[63] = 3.1416;
}

static void escribir_hoja1(DatosFourier *datos) {
    FILE *fp = fopen(CSV_HOJA1, "w");
    if (!fp) { perror("Error al crear " CSV_HOJA1); return; }
    fprintf(fp, "%s\n\n\nx,f(x)\n", func_csv_header(datos->func_type));
    for (int i = 0; i < NUM_PUNTOS; i++) {
        fprintf(fp, "%.10g,%.15g\n", datos->x_vals[i], datos->hoja1_fx[i]);
    }
    fclose(fp);
    printf("Archivo %s generado.\n", CSV_HOJA1);
}

static void escribir_hoja2(DatosFourier *datos) {
    FILE *fp = fopen(CSV_HOJA2, "w");
    if (!fp) { perror("Error al crear " CSV_HOJA2); return; }
    fprintf(fp, "\n\"F(x) = Serie de Fourier - %s\"\n", func_description(datos->func_type));
    long long cols = (datos->num_terminos < MAX_SAVED_TERMINOS) ? datos->num_terminos : MAX_SAVED_TERMINOS;
    for (long long n = 1; n <= cols; n++) fprintf(fp, ",n =");
    fprintf(fp, ",,,,\n");
    fprintf(fp, "x");
    for (long long n = 1; n <= cols; n++) fprintf(fp, ",%lld", n);
    fprintf(fp, ",a0,x,F(X),f(x)\n");
    for (int i = 0; i < NUM_PUNTOS; i++) {
        fprintf(fp, "%.10g", datos->x_vals[i]);
        for (long long n = 0; n < cols; n++) {
            fprintf(fp, ",%.15g", datos->hoja2_terminos[i][n]);
        }
        fprintf(fp, ",%.15g,%.10g,%.15g,%.15g\n",
                datos->hoja2_a0, datos->x_vals[i],
                datos->hoja2_Fx[i], datos->hoja2_fx[i]);
    }
    fclose(fp);
    printf("Archivo %s generado.\n", CSV_HOJA2);
}

int main(int argc, char **argv) {
  int func_type = FUNC_X4;
  long long num_terminos = 50;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--func") == 0 && i + 1 < argc) {
      func_type = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--terms") == 0 && i + 1 < argc) {
      num_terminos = atoll(argv[++i]);
    }
  }

  if (func_type < 0 || func_type >= NUM_FUNC_TYPES) {
    fprintf(stderr, "Error: tipo de funcion invalido: %d\n", func_type);
    return 1;
  }
  
  if (num_terminos < 1 || num_terminos > MAX_TERMINOS) {
    fprintf(stderr, "Error: numero de terminos para GPU debe estar entre 1 y %lld\n", MAX_TERMINOS);
    return 1;
  }

  DatosFourier *h_datos;  /* Datos en memoria del Host */
  DatosFourier *d_datos;  /* Datos en memoria del Device (GPU) */
  cudaError_t err;

  printf("============================================================\n");
  printf("  Serie de Fourier — CUDA\n");
  printf("  Funcion: %s  |  Terminos: %lld\n", func_description(func_type), num_terminos);
  printf("============================================================\n\n");

  /* 1. Reservar memoria en Host */
  h_datos = (DatosFourier *)malloc(sizeof(DatosFourier));
  if (!h_datos) {
    perror("Error al asignar memoria en Host");
    exit(1);
  }
  memset(h_datos, 0, sizeof(DatosFourier));
  h_datos->num_terminos = num_terminos;
  h_datos->func_type = func_type;

  /* 2. Inicializar x_vals y a0 en Host */
  generar_puntos_x(h_datos->x_vals);
  h_datos->hoja2_a0 = a0_func(func_type);

  /* 3. Reservar memoria en Device */
  err = cudaMalloc((void **)&d_datos, sizeof(DatosFourier));
  if (err != cudaSuccess) {
    fprintf(stderr, "[Error CUDA] cudaMalloc fallo: %s\n", cudaGetErrorString(err));
    free(h_datos);
    exit(1);
  }

  /* 4. Copiar datos de entrada del Host al Device */
  err = cudaMemcpy(d_datos, h_datos, sizeof(DatosFourier), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Error CUDA] cudaMemcpy Host -> Device fallo: %s\n", cudaGetErrorString(err));
    cudaFree(d_datos);
    free(h_datos);
    exit(1);
  }

  /* 5. Lanzamiento del Kernel */
  dim3 numBlocks(NUM_PUNTOS, 1);
  dim3 threadsPerBlock(256, 1);

  /* Lanzar el Kernel */
  calcular_fourier_kernel<<<numBlocks, threadsPerBlock>>>(d_datos);

  /* Esperar a que la GPU termine y sincronizar */
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "[Error CUDA] cudaDeviceSynchronize fallo: %s\n", cudaGetErrorString(err));
    cudaFree(d_datos);
    free(h_datos);
    exit(1);
  }

  /* 7. Copiar los resultados del Device de vuelta al Host */
  err = cudaMemcpy(h_datos, d_datos, sizeof(DatosFourier), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Error CUDA] cudaMemcpy Device -> Host fallo: %s\n", cudaGetErrorString(err));
    cudaFree(d_datos);
    free(h_datos);
    exit(1);
  }

  /* 8. Escribir los archivos CSV */
  escribir_hoja1(h_datos);
  escribir_hoja2(h_datos);

  /* 9. Liberar memoria */
  cudaFree(d_datos);
  free(h_datos);

  cudaDeviceReset();
  return 0;
}
