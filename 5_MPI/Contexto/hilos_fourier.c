/*
 * ============================================================================
 * fourier_hilos.c — Cálculo paralelo de la Serie de Fourier usando Hilos (pthreads)
 * ============================================================================
 *
 * Compilar:  gcc -Wall -o fourier_hilos fourier_hilos.c -lm -lpthread
 * Ejecutar:  ./fourier_hilos
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

/* ── Constantes ───────────────────────────────────────────────────────────── */
#define NUM_PUNTOS    64      /* Cantidad de puntos x (filas de datos)        */
#define NUM_TERMINOS  50      /* Cantidad de términos n de la serie           */
#define NUM_HILOS     4       /* Cantidad de hilos a crear                    */
#ifndef M_PI
#define M_PI 3.14159265358979323846 /* Fallback por si no está definido PI */
#endif

/* Archivos CSV de salida */
#define CSV_HOJA1     "hoja1.csv"
#define CSV_HOJA2     "hoja2.csv"

/* ── Estructura de Datos (Compartida por todos los hilos) ─────────────────── */
typedef struct {
    double x_vals[NUM_PUNTOS];
    double hoja1_fx[NUM_PUNTOS];
    double hoja2_terminos[NUM_PUNTOS][NUM_TERMINOS];
    double hoja2_a0;
    double hoja2_Fx[NUM_PUNTOS];
    double hoja2_fx[NUM_PUNTOS];
} DatosFourier;

/* ── Estructura de Argumentos (Para pasar a cada hilo vía apuntador) ──────── */
typedef struct {
    int id_logico;          /* Identificador lógico del hilo (0, 1, 2...) */
    int inicio;             /* Fila inicial a procesar                    */
    int fin;                /* Fila final a procesar (exclusivo)          */
    DatosFourier *datos;    /* Apuntador a la memoria compartida principal*/
} ArgumentosHilo;


/* ── Funciones Matemáticas (Reutilizadas) ─────────────────────────────────── */

/* f(x) = x⁴ − 3x */
static double f(double x) {
    return pow(x, 4) - 3.0 * x;
}

/* Término n-ésimo de Fourier */
static double termino_fourier(int n, double x) {
    double n2 = (double)n * (double)n;
    double n4 = n2 * n2;
    double signo = (n % 2 == 0) ? 1.0 : -1.0;
    double pi2 = M_PI * M_PI;

    double an = (8.0 * pi2 * n2 - 48.0) * signo / n4;
    double bn = 6.0 * signo / (double)n;

    return an * cos((double)n * x) + bn * sin((double)n * x);
}

/* Generar los 64 puntos en el intervalo [-PI, PI] */
static void generar_puntos_x(double *x_vals) {
    x_vals[0] = -3.1416;
    for (int i = 1; i <= 62; i++) {
        x_vals[i] = -3.1416 + i * 0.1;
    }
    x_vals[63] = 3.1416;
}

/* ── Escritura de CSV (Reutilizadas) ──────────────────────────────────────── */

static void escribir_hoja1(DatosFourier *datos) {
    FILE *fp = fopen(CSV_HOJA1, "w");
    if (!fp) { perror("Error al crear hoja1.csv"); return; }

    fprintf(fp, "\"f(x) = (x^4) - 3x\"\n\n\nx,f(x)\n");
    for (int i = 0; i < NUM_PUNTOS; i++) {
        fprintf(fp, "%.10g,%.15g\n", datos->x_vals[i], datos->hoja1_fx[i]);
    }
    fclose(fp);
    printf("[Proceso Padre] Archivo %s generado correctamente.\n", CSV_HOJA1);
}

static void escribir_hoja2(DatosFourier *datos) {
    FILE *fp = fopen(CSV_HOJA2, "w");
    if (!fp) { perror("Error al crear hoja2.csv"); return; }

    fprintf(fp, "\n\"F(x) = ((2*(PI^4)/5)/2) + suma((((8(PI^2)*(n^2)-48)*((-1)^n)/(n^4))*cos(nx))+(((6*((-1)^n))/n)*sin(nx))) desde n=1 hasta infinito\"\n");
    
    for (int n = 1; n <= NUM_TERMINOS; n++) fprintf(fp, ",n =");
    fprintf(fp, ",,,,\n");

    fprintf(fp, "x");
    for (int n = 1; n <= NUM_TERMINOS; n++) fprintf(fp, ",%d", n);
    fprintf(fp, ",a0,x,F(X),f(x)\n");

    for (int i = 0; i < NUM_PUNTOS; i++) {
        fprintf(fp, "%.10g", datos->x_vals[i]);
        for (int n = 0; n < NUM_TERMINOS; n++) {
            fprintf(fp, ",%.15g", datos->hoja2_terminos[i][n]);
        }
        fprintf(fp, ",%.15g,%.10g,%.15g,%.15g\n", 
                datos->hoja2_a0, datos->x_vals[i], datos->hoja2_Fx[i], datos->hoja2_fx[i]);
    }
    fclose(fp);
    printf("[Proceso Padre] Archivo %s generado correctamente.\n", CSV_HOJA2);
}


/* ── Función Principal del Hilo ───────────────────────────────────────────── */

void *rutina_hilo(void *argumentos) {
    /* 1. Recibimos los argumentos casteando el apuntador genérico (void*) */
    ArgumentosHilo *args = (ArgumentosHilo *)argumentos;
    
    /* Obtenemos el identificador real del hilo en el SO */
    pthread_t id_real = pthread_self();

    printf("[Hilo %d - ID SO: %lu] Iniciando calculo de filas [%d a %d]\n", 
           args->id_logico, (unsigned long)id_real, args->inicio, args->fin - 1);

    /* 2. Realizamos el cómputo en el bloque de filas asignado */
    for (int fila = args->inicio; fila < args->fin; fila++) {
        double x = args->datos->x_vals[fila];

        /* Hoja 1: f(x) */
        args->datos->hoja1_fx[fila] = f(x);

        /* Hoja 2: Serie de Fourier */
        double suma = 0.0;
        for (int n = 1; n <= NUM_TERMINOS; n++) {
            double term = termino_fourier(n, x);
            args->datos->hoja2_terminos[fila][n - 1] = term;
            suma += term;
        }

        /* Resultados finales por fila */
        args->datos->hoja2_Fx[fila] = (args->datos->hoja2_a0 / 2.0) + suma;
        args->datos->hoja2_fx[fila] = f(x);
    }

    printf("[Hilo %d] Trabajo finalizado.\n", args->id_logico);

    /* 3. Terminamos la ejecución del hilo */
    pthread_exit(NULL);
}


/* ── Proceso Padre (Main) ─────────────────────────────────────────────────── */

int main(void) {
    pthread_t identificadores[NUM_HILOS];
    ArgumentosHilo args_hilos[NUM_HILOS];
    
    /* Asignamos memoria en el heap para la estructura principal de datos */
    DatosFourier *datos = (DatosFourier *)malloc(sizeof(DatosFourier));
    if (datos == NULL) {
        perror("Error asignando memoria para los datos");
        exit(EXIT_FAILURE);
    }

    printf("============================================================\n");
    printf("  Serie de Fourier — Cómputo Paralelo con Hilos (pthreads)\n");
    printf("============================================================\n\n");

    /* Inicializamos los datos compartidos */
    generar_puntos_x(datos->x_vals);
    datos->hoja2_a0 = (2.0 * pow(M_PI, 4)) / 5.0;

    int filas_por_hilo = NUM_PUNTOS / NUM_HILOS;
    int filas_restantes = NUM_PUNTOS % NUM_HILOS;

/* ─── Creación de los hilos ─── */
    printf("[Proceso Padre] Creando %d hilos...\n\n", NUM_HILOS);
    for (int i = 0; i < NUM_HILOS; i++) {
        /* Preparamos los parámetros que le pasaremos al hilo */
        args_hilos[i].id_logico = i + 1;
        args_hilos[i].inicio = i * filas_por_hilo;
        args_hilos[i].fin = args_hilos[i].inicio + filas_por_hilo;
        args_hilos[i].datos = datos; /* Todos apuntan a la misma estructura */

        /* El último hilo toma el residuo de filas si la división no es exacta */
        if (i == NUM_HILOS - 1) {
            args_hilos[i].fin += filas_restantes;
        }

        /* Se crea el hilo pasando el struct 'ArgumentosHilo' por referencia */
        if (pthread_create(&identificadores[i], NULL, rutina_hilo, (void *)&args_hilos[i]) != 0) {
            perror("Error al crear el hilo");
            free(datos);
            exit(EXIT_FAILURE);
        }
    }

/* ─── Sincronización: Esperar a que terminen los hilos ─── */
    for (int i = 0; i < NUM_HILOS; i++) {
        pthread_join(identificadores[i], NULL);
    }

    printf("\n[Proceso Padre] Todos los hilos han finalizado. Escribiendo resultados...\n\n");

    /* Escribir los resultados en disco */
    escribir_hoja1(datos);
    escribir_hoja2(datos);

    /* Liberar memoria */
    free(datos);

    printf("\n[Proceso Padre] Programa finalizado con éxito.\n");
    return 0;
}