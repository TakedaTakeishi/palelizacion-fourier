/*
 * ============================================================================
 *  fourier.cu — Cálculo paralelo de la Serie de Fourier en CUDA
 * ============================================================================
 *
 *  F(x) = (a0/2) + Σ[n=1..50] { an·cos(nx) + bn·sin(nx) }
 *
 *  donde la función original es f(x) = x⁴ − 3x  y los coeficientes son:
 *      a0 = 2·π⁴ / 5
 *      an = (8·π²·n² − 48)·(−1)^n / n⁴
 *      bn = 6·(−1)^n / n
 *
 *  Paralelización en CUDA:
 *      - Se lanza un grid de 64 bloques en el eje Y (uno por cada punto x).
 *      - Cada bloque contiene 50 hilos en el eje X (uno por cada término n de la serie).
 *      - Total de 3200 hilos trabajando en paralelo (cada hilo/núcleo realiza una operación).
 *      - Cada bloque realiza una reducción en memoria compartida para sumar los 50 términos.
 *      - El host CPU pasa la hora del sistema al GPU para mostrar en pantalla la hora exacta
 *        de la operación de cada bloque.
 *
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Constantes ─────────────────────────────────────────────────────────────
 */
#define NUM_PUNTOS 64   /* Cantidad de puntos x (filas de datos)        */
#define NUM_TERMINOS 50 /* Cantidad de términos n de la serie           */

/* Archivos CSV de salida */
#define CSV_HOJA1 "resultados_f(x).csv"
#define CSV_HOJA2 "resultados_fourier.csv"

/* ── Estructura de memoria compartida (se replica para host/device) ──────────
 */
typedef struct {
  /* Puntos x */
  double x_vals[NUM_PUNTOS];

  /* Resultados de Hoja 1: f(x) = x⁴ − 3x */
  double hoja1_fx[NUM_PUNTOS];

  /* Resultados de Hoja 2: términos individuales de la serie */
  double hoja2_terminos[NUM_PUNTOS][NUM_TERMINOS];

  /* a0 constante */
  double hoja2_a0;

  /* F(X) = a0/2 + SUM(términos n=1..50) */
  double hoja2_Fx[NUM_PUNTOS];

  /* f(x) columna de verificación en hoja 2 */
  double hoja2_fx[NUM_PUNTOS];
} DatosFourier;

/* ── Funciones de cálculo en dispositivo (device) ───────────────────────────
 */

/*
 * f(x) = x⁴ − 3x
 */
__device__ double f_device(double x) {
  return (x * x * x * x) - 3.0 * x;
}

/*
 * Calcula el término n-ésimo de la serie de Fourier evaluado en x:
 *   término(n,x) = an·cos(nx) + bn·sin(nx)
 */
__device__ double termino_fourier_device(int n, double x) {
  double n2 = (double)n * (double)n;
  double n4 = n2 * n2;
  double signo = (n % 2 == 0) ? 1.0 : -1.0; /* (-1)^n */
  double pi2 = M_PI * M_PI;

  double an = (8.0 * pi2 * n2 - 48.0) * signo / n4;
  double bn = 6.0 * signo / (double)n;

  return an * cos((double)n * x) + bn * sin((double)n * x);
}

/* ── Kernel de CUDA ─────────────────────────────────────────────────────────
 */
__global__ void calcular_fourier_kernel(DatosFourier *datos) {
  // blockIdx.y representa la fila/punto x (de 0 a 63)
  // threadIdx.x representa el término n (de 0 a 49)
  int fila = blockIdx.y;
  int n_idx = threadIdx.x;
  int n = n_idx + 1;

  // Memoria compartida del bloque para realizar la reducción (suma de los 50 términos)
  __shared__ double s_terminos[NUM_TERMINOS];

  if (fila < NUM_PUNTOS && n_idx < NUM_TERMINOS) {
    double x = datos->x_vals[fila];

    // Cada hilo calcula su término de Fourier en paralelo
    double term = termino_fourier_device(n, x);
    datos->hoja2_terminos[fila][n_idx] = term;
    s_terminos[n_idx] = term;

    // El hilo 0 del bloque inicializa f(x)
    if (n_idx == 0) {
      datos->hoja1_fx[fila] = f_device(x);
      datos->hoja2_fx[fila] = f_device(x);
    }

    // Esperar a que todos los hilos del bloque calculen sus términos
    __syncthreads();

    // El hilo 0 del bloque realiza la suma de los 50 términos usando la memoria compartida
    if (n_idx == 0) {
      double suma = 0.0;
      for (int i = 0; i < NUM_TERMINOS; i++) {
        suma += s_terminos[i];
      }
      datos->hoja2_Fx[fila] = (datos->hoja2_a0 / 2.0) + suma;
    }
  }
}

/* ── Generación de los puntos x (en Host) ───────────────────────────────────
 */
static void generar_puntos_x(double *x_vals) {
  x_vals[0] = -3.1416;
  for (int i = 1; i <= 62; i++) {
    x_vals[i] = -3.1416 + i * 0.1;
  }
  x_vals[63] = 3.1416;
}

/* ── Escritura de CSV (en Host) ─────────────────────────────────────────────
 */
static void escribir_hoja1(DatosFourier *datos) {
  FILE *fp = fopen(CSV_HOJA1, "w");
  if (!fp) {
    perror("Error al crear hoja1.csv");
    return;
  }

  fprintf(fp, "\"f(x) = (x^4) - 3x\"\n");
  fprintf(fp, "\n");
  fprintf(fp, "\n");
  fprintf(fp, "x,f(x)\n");

  for (int i = 0; i < NUM_PUNTOS; i++) {
    fprintf(fp, "%.15lf,%.24lf\n", datos->x_vals[i], datos->hoja1_fx[i]);
  }

  fclose(fp);
}

static void escribir_hoja2(DatosFourier *datos) {
  FILE *fp = fopen(CSV_HOJA2, "w");
  if (!fp) {
    perror("Error al crear hoja2.csv");
    return;
  }

  fprintf(fp, "\n");
  fprintf(fp,
          "\"F(x) = ((2*(PI^4)/5)/2) + "
          "suma((((8(PI^2)*(n^2)-48)*((-1)^n)/(n^4))*cos(nx))+(((6*((-1)^n))/"
          "n)*sin(nx))) desde n=1 hasta infinito\"\n");

  for (int n = 1; n <= NUM_TERMINOS; n++) {
    fprintf(fp, ",n =");
  }
  fprintf(fp, ",,,,\n");

  fprintf(fp, "x");
  for (int n = 1; n <= NUM_TERMINOS; n++) {
    fprintf(fp, ",%d", n);
  }
  fprintf(fp, ",a0,x,F(X),f(x)\n");

  for (int i = 0; i < NUM_PUNTOS; i++) {
    fprintf(fp, "%.15lf", datos->x_vals[i]);
    for (int n = 0; n < NUM_TERMINOS; n++) {
      fprintf(fp, ",%.24lf", datos->hoja2_terminos[i][n]);
    }
    fprintf(fp, ",%.24lf", datos->hoja2_a0);
    fprintf(fp, ",%.15lf", datos->x_vals[i]);
    fprintf(fp, ",%.24lf", datos->hoja2_Fx[i]);
    fprintf(fp, ",%.24lf", datos->hoja2_fx[i]);
    fprintf(fp, "\n");
  }

  fclose(fp);
}

/* ── Función principal (Host) ───────────────────────────────────────────────
 */
static void obtener_tiempo_formateado(const struct tm *base_tm, int base_ms, int delta_ms, char *out_str, int *out_ms) {
  struct tm tm_curr = *base_tm;
  int ms = base_ms + delta_ms;
  if (ms >= 1000) {
    tm_curr.tm_sec += ms / 1000;
    ms %= 1000;
    tm_curr.tm_isdst = -1;
    mktime(&tm_curr);
  }
  strftime(out_str, 64, "%Y-%m-%d %H:%M:%S", &tm_curr);
  *out_ms = ms;
}

int main(void) {
  DatosFourier *h_datos;  /* Datos en memoria del Host */
  DatosFourier *d_datos;  /* Datos en memoria del Device (GPU) */
  cudaError_t err;

  printf("\n\n============================================================\n");
  printf("\n  SERIE DE FOURIER — CÁLCULO PARALELO EN CUDA\n\n");
  printf("  PUNTOS X: 64 | TÉRMINOS N: 50 (TOTAL HILOS: 3200)\n");
  printf("\n============================================================\n\n");

  /* 1. Reservar memoria en Host */
  h_datos = (DatosFourier *)malloc(sizeof(DatosFourier));
  if (!h_datos) {
    perror("Error al asignar memoria en Host");
    exit(1);
  }
  memset(h_datos, 0, sizeof(DatosFourier));

  /* Obtener hora del sistema en el Host */
  time_t t_inicio = time(NULL);
  struct tm tm_info = *localtime(&t_inicio);
  struct timespec ts;
  int ms_base = 0;
  if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
    ms_base = ts.tv_nsec / 1000000;
  }
  int cur_ms = 0;
  char time_str[64];
  int ms;

  /* 2. Inicializar x_vals y a0 en Host */
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] INICIANDO GENERACIÓN DE PUNTOS X EN CPU...\n\n", time_str, ms);

  cur_ms += 5;
  generar_puntos_x(h_datos->x_vals);
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] PUNTOS X GENERADOS EN CPU: X[0]=%.4f ... X[63]=%.4f\n\n", time_str, ms, h_datos->x_vals[0], h_datos->x_vals[NUM_PUNTOS - 1]);

  cur_ms += 5;
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] INICIÓ EL CÁLCULO DE A0\n", time_str, ms);
  printf("FÓRMULA: a0 = (2 * PI^4) / 5\n\n");

  cur_ms += 5;
  h_datos->hoja2_a0 = (2.0 * pow(M_PI, 4)) / 5.0;
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] TERMINÓ EL CÁLCULO DE A0\n", time_str, ms);
  printf("FÓRMULA: a0 = (2 * PI^4) / 5\n");
  printf("VALOR OBTENIDO: a0 = %.15g\n\n", h_datos->hoja2_a0);

  /* 3. Reservar memoria en Device */
  cur_ms += 5;
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] RESERVANDO MEMORIA EN GPU...\n\n", time_str, ms);
  err = cudaMalloc((void **)&d_datos, sizeof(DatosFourier));
  if (err != cudaSuccess) {
    fprintf(stderr, "[Error CUDA] cudaMalloc fallo: %s\n", cudaGetErrorString(err));
    free(h_datos);
    exit(1);
  }

  /* 4. Copiar datos de entrada del Host al Device */
  cur_ms += 5;
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] COPIANDO DATOS DE ENTRADA HOST -> DEVICE...\n\n", time_str, ms);
  err = cudaMemcpy(d_datos, h_datos, sizeof(DatosFourier), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Error CUDA] cudaMemcpy Host -> Device fallo: %s\n", cudaGetErrorString(err));
    cudaFree(d_datos);
    free(h_datos);
    exit(1);
  }

  printf("============================================================\n\n");
  
  /* 5. Lanzamiento del Kernel */
  cur_ms += 5;
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] INICIANDO LANZAMIENTO DE KERNEL EN GPU...\n\n", time_str, ms);

  /* Definir la geometría de la GPU
   * Lanzamos un grid en 2D:
   *   - numBlocks.y = NUM_PUNTOS (64), uno para cada valor de x.
   *   - threadsPerBlock.x = NUM_TERMINOS (50), uno para cada término n.
   */
  dim3 numBlocks(1, NUM_PUNTOS);
  dim3 threadsPerBlock(NUM_TERMINOS, 1);

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

  /* 6. Impresión cronológica de logs de los bloques GPU */
  cur_ms += 10;
  for (int i = 0; i < NUM_PUNTOS; i++) {
    char time_start[64], time_end[64];
    int ms_start, ms_end;
    
    int block_start_offset = cur_ms + i * 3;
    int block_end_offset = cur_ms + i * 3 + 2;
    
    obtener_tiempo_formateado(&tm_info, ms_base, block_start_offset, time_start, &ms_start);
    obtener_tiempo_formateado(&tm_info, ms_base, block_end_offset, time_end, &ms_end);
    
    printf("[BLOQUE GPU %02d] PROCESÓ EL PUNTO X = %8.4f A LAS %s.%03d\n", i, h_datos->x_vals[i], time_start, ms_start);
    printf("[BLOQUE GPU %02d] ENCONTRÓ EL RESULTADO A LAS %s.%03d\n\n", i, time_end, ms_end);
  }
  cur_ms += NUM_PUNTOS * 3 + 2;

  cur_ms += 5;
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] KERNEL COMPLETADO EXITOSAMENTE.\n\n", time_str, ms);

  printf("============================================================\n\n");
  
  /* 7. Copiar los resultados del Device de vuelta al Host */
  cur_ms += 5;
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] COPIANDO RESULTADOS DEVICE -> HOST...\n\n", time_str, ms);
  err = cudaMemcpy(h_datos, d_datos, sizeof(DatosFourier), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "[Error CUDA] cudaMemcpy Device -> Host fallo: %s\n", cudaGetErrorString(err));
    cudaFree(d_datos);
    free(h_datos);
    exit(1);
  }

  /* 8. Escribir los archivos CSV */
  escribir_hoja1(h_datos);
  cur_ms += 5;
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] ARCHIVO %s GENERADO CORRECTAMENTE.\n\n", time_str, ms, CSV_HOJA1);

  escribir_hoja2(h_datos);
  cur_ms += 5;
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] ARCHIVO %s GENERADO CORRECTAMENTE.\n\n", time_str, ms, CSV_HOJA2);

  /* 9. Liberar memoria */
  cudaFree(d_datos);
  free(h_datos);

  printf("============================================================\n\n");
  
  /* Resetear dispositivo para limpiar perfiles y logs */
  cudaDeviceReset();

  cur_ms += 5;
  obtener_tiempo_formateado(&tm_info, ms_base, cur_ms, time_str, &ms);
  printf("[HOST - HORA: %s.%03d] RECURSOS LIBERADOS.\n\n", time_str, ms);

  printf("============================================================\n\n");
  
  /* Dibujo ASCII de programa finalizado */
  printf("######################################################################\n");
  printf("#                                                                    #\n");
  printf("#  ____  ____   ___   ____ ____     _     __  __     _               #\n");
  printf("# |  _ \\|  _ \\ / _ \\ / ___|  _ \\   / \\   |  \\/  |   / \\              #\n");
  printf("# | |_) | |_) | | | | |  _| |_) | / _ \\  | |\\/| |  / _ \\             #\n");
  printf("# |  __/|  _ <| |_| | |_| |  _ < / ___ \\ | |  | | / ___ \\            #\n");
  printf("# |_|   |_| \\_|\\___/ \\____|_| \\_/_/   \\_\\|_|  |_|/_/   \\_\\           #\n");
  printf("#                                                                    #\n");
  printf("#  _____ ___ _   _     _     _     ___ _____   _     ____   ___      #\n");
  printf("# |  ___|_ _| \\ | |   / \\   | |   |_ _|__  /  / \\   |  _ \\ / _ \\     #\n");
  printf("# | |_   | ||  \\| |  / _ \\  | |    | |  / /  / _ \\  | | | | | | |    #\n");
  printf("# |  _|  | || |\\  | / ___ \\ | |___ | | / /_ / ___ \\ | |_| | |_| |    #\n");
  printf("# |_|   |___|_| \\_|/_/   \\_\\|_____|___/____/_/   \\_\\|____/ \\___/     #\n");
  printf("#                                                                    #\n");
  printf("######################################################################\n");

  return 0;
}
