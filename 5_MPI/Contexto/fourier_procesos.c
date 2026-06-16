/*
 * ============================================================================
 *  fourier.c — Cálculo paralelo de la Serie de Fourier
 * ============================================================================
 *
 *  F(x) = (a0/2) + Σ[n=1..50] { an·cos(nx) + bn·sin(nx) }
 *
 *  donde la función original es f(x) = x⁴ − 3x  y los coeficientes son:
 *      a0 = 2·π⁴ / 5
 *      an = (8·π²·n² − 48)·(−1)^n / n⁴
 *      bn = 6·(−1)^n / n
 *
 *  Paralelización: se divide el eje x (64 puntos) entre NUM_HIJOS procesos.
 *  Cada proceso hijo calcula, para su bloque de filas, todos los 50 términos,
 *  a0, F(X) y f(x), y los deposita en memoria compartida.
 *
 *  Mecanismos IPC utilizados:
 *      - fork()              → creación de procesos hijos
 *      - shmget / shmat      → memoria compartida System V
 *      - semget / semop      → semáforos System V
 *
 *  Compilar:  gcc -Wall -o fourier fourier.c -lm
 *  Ejecutar:  ./fourier
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <time.h>

/* ── Constantes ───────────────────────────────────────────────────────────── */

#define NUM_PUNTOS    64      /* Cantidad de puntos x (filas de datos)        */
#define NUM_TERMINOS  50      /* Cantidad de términos n de la serie           */
#define NUM_HIJOS     4       /* Cantidad de procesos hijos a crear           */
#define PERMISOS      0666    /* Permisos para recursos IPC                   */

/* Identificadores para generar llaves con ftok */
#define ARCHIVO_LLAVE "fourier.c"
#define ID_MEMORIA    'M'
#define ID_SEM_BASE   'S'     /* Los semáforos se generan con 'S'+i           */

/* Archivos CSV de salida */
#define CSV_HOJA1     "hoja1.csv"
#define CSV_HOJA2     "hoja2.csv"

/* ── Estructura de memoria compartida ─────────────────────────────────────── */

/*
 * Almacena TODOS los resultados que los hijos calculan.
 * El padre los lee después para escribir los CSV.
 *
 * Diseño de columnas en hoja2 (por fila i):
 *   terminos[i][0..49]  → columnas B..AY  (n=1..50)
 *   a0                  → columna  AZ      (constante, igual para todas las filas)
 *   Fx[i]               → columna  BB      (= a0/2 + SUM(terminos))
 *   fx[i]               → columna  BC      (= x⁴ − 3x, verificación)
 */
typedef struct {
    /* Puntos x compartidos (los llena el padre antes de hacer fork) */
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

/* ── union semun (necesaria en algunos sistemas para semctl) ──────────────── */

#if defined(__GNU_LIBRARY__) && !defined(_SEM_SEMUN_UNDEFINED)
    /* union semun ya está definida */
#else
union semun {
    int val;
    struct semid_ds *buf;
    unsigned short int *array;
    struct seminfo *__buf;
};
#endif

/* ── Funciones auxiliares para semáforos ───────────────────────────────────── */

/*
 * Crea un semáforo System V con la llave dada y lo inicializa a valor_inicial.
 * Retorna el identificador del semáforo.
 */
static int Crea_semaforo(key_t llave, int valor_inicial) {
    int semid = semget(llave, 1, IPC_CREAT | PERMISOS);
    if (semid == -1) {
        perror("Error al crear semaforo");
        exit(1);
    }
    union semun arg;
    arg.val = valor_inicial;
    if (semctl(semid, 0, SETVAL, arg) == -1) {
        perror("Error al inicializar semaforo");
        exit(1);
    }
    return semid;
}

/*
 * Operación down (P / wait): decrementa el semáforo.
 * Si el valor es 0, el proceso se bloquea hasta que alguien haga up.
 */
static void down(int semid) {
    struct sembuf op = {0, -1, 0};
    if (semop(semid, &op, 1) == -1) {
        perror("Error en operacion down");
    }
}

/*
 * Operación up (V / signal): incrementa el semáforo.
 * Desbloquea a un proceso que esté esperando en down.
 */
static void up(int semid) {
    struct sembuf op = {0, 1, 0};
    if (semop(semid, &op, 1) == -1) {
        perror("Error en operacion up");
    }
}

/* ── Funciones de cálculo ─────────────────────────────────────────────────── */

/*
 * f(x) = x⁴ − 3x
 * La función original cuya serie de Fourier aproximamos.
 */
static double f(double x) {
    return pow(x, 4) - 3.0 * x;
}

/*
 * Calcula el término n-ésimo de la serie de Fourier evaluado en x:
 *   término(n,x) = an·cos(nx) + bn·sin(nx)
 *
 * donde:
 *   an = (8·π²·n² − 48)·(−1)^n / n⁴
 *   bn = 6·(−1)^n / n
 */
static double termino_fourier(int n, double x) {
    double n2 = (double)n * (double)n;
    double n4 = n2 * n2;
    double signo = (n % 2 == 0) ? 1.0 : -1.0;   /* (-1)^n */
    double pi2 = M_PI * M_PI;

    double an = (8.0 * pi2 * n2 - 48.0) * signo / n4;
    double bn = 6.0 * signo / (double)n;

    return an * cos((double)n * x) + bn * sin((double)n * x);
}

/* ── Generación de los puntos x ───────────────────────────────────────────── */

/*
 * Genera los 64 puntos x tal como están en el Excel:
 *   x[0]  = −3.1416   (≈ −π)
 *   x[1]  = −3.0416
 *   x[2]  = −2.9416
 *   ...                (paso 0.1)
 *   x[62] = 3.0584
 *   x[63] = 3.1416    (≈ π)
 *
 * Son 64 puntos: el primero es −π, luego 62 puntos con paso 0.1, y el último es π.
 */
static void generar_puntos_x(double *x_vals) {
    x_vals[0] = -3.1416;
    for (int i = 1; i <= 62; i++) {
        x_vals[i] = -3.1416 + i * 0.1;
    }
    x_vals[63] = 3.1416;
}

/* ── Escritura de CSV ─────────────────────────────────────────────────────── */

/*
 * Escribe hoja1.csv reproduciendo la estructura del Excel:
 *
 *   Fila 1: "f(x) = (x^4) - 3x"
 *   Fila 2: (vacía)
 *   Fila 3: (vacía)
 *   Fila 4: "x","f(x)"
 *   Filas 5-68: datos
 */
static void escribir_hoja1(DatosFourier *datos) {
    FILE *fp = fopen(CSV_HOJA1, "w");
    if (!fp) {
        perror("Error al crear hoja1.csv");
        return;
    }

    /* Encabezados (reproduciendo estructura del Excel) */
    fprintf(fp, "\"f(x) = (x^4) - 3x\"\n");
    fprintf(fp, "\n");
    fprintf(fp, "\n");
    fprintf(fp, "x,f(x)\n");

    /* Datos */
    for (int i = 0; i < NUM_PUNTOS; i++) {
        fprintf(fp, "%.10g,%.15g\n", datos->x_vals[i], datos->hoja1_fx[i]);
    }

    fclose(fp);
    printf("[Padre] Archivo %s generado correctamente.\n", CSV_HOJA1);
}

/*
 * Escribe hoja2.csv reproduciendo la estructura del Excel:
 *
 *   Fila 1: (vacía)
 *   Fila 2: "F(x) = ..."
 *   Fila 3: "",n =,n =,n =,...
 *   Fila 4: x,1,2,3,...,49,50,a0,x,F(X),f(x)
 *   Filas 5-68: datos
 *
 * Columnas:
 *   A   = x
 *   B-AY = términos n=1..50
 *   AZ  = a0
 *   BA  = x  (repetido)
 *   BB  = F(X)
 *   BC  = f(x)
 */
static void escribir_hoja2(DatosFourier *datos) {
    FILE *fp = fopen(CSV_HOJA2, "w");
    if (!fp) {
        perror("Error al crear hoja2.csv");
        return;
    }

    /* ─ Fila 1: vacía ─ */
    fprintf(fp, "\n");

    /* ─ Fila 2: título de la serie ─ */
    fprintf(fp, "\"F(x) = ((2*(PI^4)/5)/2) + suma((((8(PI^2)*(n^2)-48)*((-1)^n)/(n^4))*cos(nx))+(((6*((-1)^n))/n)*sin(nx))) desde n=1 hasta infinito\"\n");

    /* ─ Fila 3: etiquetas "n =" ─ */
    /* columna A vacía — no se imprime nada antes de la primera coma */
    for (int n = 1; n <= NUM_TERMINOS; n++) {
        fprintf(fp, ",n =");
    }
    /* columnas AZ, BA, BB, BC vacías en fila 3 */
    fprintf(fp, ",,,,\n");

    /* ─ Fila 4: encabezados de columna ─ */
    fprintf(fp, "x");
    for (int n = 1; n <= NUM_TERMINOS; n++) {
        fprintf(fp, ",%d", n);
    }
    fprintf(fp, ",a0,x,F(X),f(x)\n");

    /* ─ Filas de datos (5-68 en el Excel) ─ */
    for (int i = 0; i < NUM_PUNTOS; i++) {
        /* Columna A: x */
        fprintf(fp, "%.10g", datos->x_vals[i]);

        /* Columnas B-AY: términos n=1..50 */
        for (int n = 0; n < NUM_TERMINOS; n++) {
            fprintf(fp, ",%.15g", datos->hoja2_terminos[i][n]);
        }

        /* Columna AZ: a0 */
        fprintf(fp, ",%.15g", datos->hoja2_a0);

        /* Columna BA: x (repetido) */
        fprintf(fp, ",%.10g", datos->x_vals[i]);

        /* Columna BB: F(X) */
        fprintf(fp, ",%.15g", datos->hoja2_Fx[i]);

        /* Columna BC: f(x) */
        fprintf(fp, ",%.15g", datos->hoja2_fx[i]);

        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("[Padre] Archivo %s generado correctamente.\n", CSV_HOJA2);
}

/* ── Función principal ────────────────────────────────────────────────────── */

int main(void) {
    key_t llave_memoria;
    int shmid;
    DatosFourier *datos;
    int sem_ids[NUM_HIJOS];     /* Un semáforo por hijo */
    key_t llave_sem;
    pid_t pids[NUM_HIJOS];
    int i;

    printf("============================================================\n");
    printf("  Serie de Fourier — Cálculo Paralelo con fork()\n");
    printf("  Procesos hijos: %d | Puntos x: %d | Términos n: %d\n",
           NUM_HIJOS, NUM_PUNTOS, NUM_TERMINOS);
    printf("============================================================\n\n");

    /* ──────────────────────────────────────────────────────────────────────
     * 1. Crear memoria compartida
     * ────────────────────────────────────────────────────────────────────── */
    llave_memoria = ftok(ARCHIVO_LLAVE, ID_MEMORIA);
    if (llave_memoria == -1) {
        perror("Error en ftok para memoria");
        exit(1);
    }

    shmid = shmget(llave_memoria, sizeof(DatosFourier), IPC_CREAT | PERMISOS);
    if (shmid == -1) {
        perror("Error al crear memoria compartida");
        exit(1);
    }

    datos = (DatosFourier *)shmat(shmid, NULL, 0);
    if (datos == (void *)-1) {
        perror("Error al adjuntar memoria compartida");
        shmctl(shmid, IPC_RMID, NULL);
        exit(1);
    }

    /* Limpiar la memoria compartida */
    memset(datos, 0, sizeof(DatosFourier));

    printf("[Padre PID=%d] Memoria compartida creada (shmid=%d, tamanio=%lu bytes)\n",
           getpid(), shmid, (unsigned long)sizeof(DatosFourier));

    /* ──────────────────────────────────────────────────────────────────────
     * 2. Llenar los puntos x y calcular a0
     * ────────────────────────────────────────────────────────────────────── */
    generar_puntos_x(datos->x_vals);
    datos->hoja2_a0 = (2.0 * pow(M_PI, 4)) / 5.0;

    printf("[Padre] Puntos x generados: x[0]=%.4f ... x[63]=%.4f\n",
           datos->x_vals[0], datos->x_vals[NUM_PUNTOS - 1]);
    printf("[Padre] a0 = %.15g\n", datos->hoja2_a0);

    /* ──────────────────────────────────────────────────────────────────────
     * 3. Crear semáforos (uno por hijo, inicializados en 0 = bloqueados)
     * ────────────────────────────────────────────────────────────────────── */
    for (i = 0; i < NUM_HIJOS; i++) {
        llave_sem = ftok(ARCHIVO_LLAVE, ID_SEM_BASE + i);
        if (llave_sem == -1) {
            perror("Error en ftok para semaforo");
            exit(1);
        }
        sem_ids[i] = Crea_semaforo(llave_sem, 0);
    }

    printf("[Padre] %d semaforos creados.\n\n", NUM_HIJOS);

    /* ──────────────────────────────────────────────────────────────────────
     * 4. Calcular la distribución de filas entre hijos
     *    Se divide el intervalo [0, NUM_PUNTOS) en NUM_HIJOS bloques.
     *    Si no divide exactamente, el último hijo toma las filas restantes.
     * ────────────────────────────────────────────────────────────────────── */
    int filas_por_hijo = NUM_PUNTOS / NUM_HIJOS;
    int filas_restantes = NUM_PUNTOS % NUM_HIJOS;

    /* ──────────────────────────────────────────────────────────────────────
     * 5. Crear los procesos hijos con fork()
     * ────────────────────────────────────────────────────────────────────── */
    for (i = 0; i < NUM_HIJOS; i++) {
        pids[i] = fork();

        if (pids[i] < 0) {
            perror("Error al crear proceso hijo con fork()");
            /* Limpiar recursos antes de salir */
            shmdt(datos);
            shmctl(shmid, IPC_RMID, NULL);
            for (int j = 0; j <= i; j++) {
                semctl(sem_ids[j], 0, IPC_RMID);
            }
            exit(1);
        }

        if (pids[i] == 0) {
            /* ═══════════════════════════════════════════════════════════════
             *  CÓDIGO DEL PROCESO HIJO
             * ═══════════════════════════════════════════════════════════════ */

            /* Calcular el rango de filas que le toca a este hijo */
            int inicio = i * filas_por_hijo;
            int fin    = inicio + filas_por_hijo;
            /* El último hijo absorbe las filas restantes */
            if (i == NUM_HIJOS - 1) {
                fin += filas_restantes;
            }

            /* Obtener marca de tiempo */
            time_t t = time(NULL);
            struct tm *tm_info = localtime(&t);
            char timestamp[64];
            strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);

            printf("[Hijo %d, PID=%d] Esperando semaforo... (%s)\n",
                   i, getpid(), timestamp);

            /* Esperar a que el padre libere el semáforo */
            down(sem_ids[i]);

            t = time(NULL);
            tm_info = localtime(&t);
            strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);

            printf("[Hijo %d, PID=%d] Iniciando calculo de filas [%d, %d) (%s)\n",
                   i, getpid(), inicio, fin, timestamp);

            /* ─── Calcular para cada fila asignada ─── */
            for (int fila = inicio; fila < fin; fila++) {
                double x = datos->x_vals[fila];

                /* Hoja 1: f(x) = x⁴ − 3x */
                datos->hoja1_fx[fila] = f(x);

                /* Hoja 2: cada término de la serie */
                double suma = 0.0;
                for (int n = 1; n <= NUM_TERMINOS; n++) {
                    double term = termino_fourier(n, x);
                    datos->hoja2_terminos[fila][n - 1] = term;
                    suma += term;
                }

                /* F(X) = a0/2 + SUM(términos) */
                datos->hoja2_Fx[fila] = (datos->hoja2_a0 / 2.0) + suma;

                /* f(x) para columna de verificación */
                datos->hoja2_fx[fila] = f(x);
            }

            t = time(NULL);
            tm_info = localtime(&t);
            strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", tm_info);

            printf("[Hijo %d, PID=%d] Calculo completado (%d filas). (%s)\n",
                   i, getpid(), fin - inicio, timestamp);

            /* Desconectarse de la memoria compartida y salir */
            shmdt(datos);
            _exit(0);
        }
    }

    /* ══════════════════════════════════════════════════════════════════════
     *  CÓDIGO DEL PROCESO PADRE (continúa aquí)
     * ══════════════════════════════════════════════════════════════════════ */

    /* ──────────────────────────────────────────────────────────────────────
     * 6. Liberar los semáforos para que los hijos comiencen a trabajar
     * ────────────────────────────────────────────────────────────────────── */
    printf("[Padre] Liberando semaforos para que los hijos inicien...\n\n");
    for (i = 0; i < NUM_HIJOS; i++) {
        up(sem_ids[i]);
    }

    /* ──────────────────────────────────────────────────────────────────────
     * 7. Esperar a que TODOS los hijos terminen
     * ────────────────────────────────────────────────────────────────────── */
    for (i = 0; i < NUM_HIJOS; i++) {
        int estado;
        pid_t terminado = waitpid(pids[i], &estado, 0);
        if (terminado == -1) {
            perror("Error en waitpid");
        } else if (WIFEXITED(estado) && WEXITSTATUS(estado) == 0) {
            printf("[Padre] Hijo PID=%d termino exitosamente.\n", terminado);
        } else {
            fprintf(stderr, "[Padre] Hijo PID=%d termino con error (estado=%d).\n",
                    terminado, WEXITSTATUS(estado));
        }
    }

    printf("\n[Padre] Todos los hijos finalizaron. Generando archivos CSV...\n\n");

    /* ──────────────────────────────────────────────────────────────────────
     * 8. Escribir los archivos CSV
     * ────────────────────────────────────────────────────────────────────── */
    escribir_hoja1(datos);
    escribir_hoja2(datos);

    /* ──────────────────────────────────────────────────────────────────────
     * 9. Liberar recursos IPC
     * ────────────────────────────────────────────────────────────────────── */
    shmdt(datos);
    shmctl(shmid, IPC_RMID, NULL);
    for (i = 0; i < NUM_HIJOS; i++) {
        semctl(sem_ids[i], 0, IPC_RMID);
    }

    printf("\n[Padre] Recursos IPC liberados. Programa finalizado.\n");
    printf("============================================================\n");

    return 0;
}
