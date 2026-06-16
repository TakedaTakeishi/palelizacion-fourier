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
#include "../common/fourier_core.h"

#define NUM_HIJOS     4
#define PERMISOS      0666
#define ARCHIVO_LLAVE "fourier.c"
#define ID_MEMORIA    'M'
#define ID_SEM_BASE   'S'
#define CSV_HOJA1     "hoja1.csv"
#define CSV_HOJA2     "hoja2.csv"

#if defined(__GNU_LIBRARY__) && !defined(_SEM_SEMUN_UNDEFINED)
#else
union semun {
    int val;
    struct semid_ds *buf;
    unsigned short int *array;
    struct seminfo *__buf;
};
#endif

static int Crea_semaforo(key_t llave, int valor_inicial) {
    int semid = semget(llave, 1, IPC_CREAT | PERMISOS);
    if (semid == -1) { perror("Error creando semaforo"); exit(1); }
    union semun arg;
    arg.val = valor_inicial;
    if (semctl(semid, 0, SETVAL, arg) == -1) { perror("Error init semaforo"); exit(1); }
    return semid;
}

static void down(int semid) {
    struct sembuf op = {0, -1, 0};
    if (semop(semid, &op, 1) == -1) perror("Error en down");
}

static void up(int semid) {
    struct sembuf op = {0, 1, 0};
    if (semop(semid, &op, 1) == -1) perror("Error en up");
}

static void generar_puntos_x(double *x_vals) {
    x_vals[0] = -3.1416;
    for (int i = 1; i <= 62; i++)
        x_vals[i] = -3.1416 + i * 0.1;
    x_vals[63] = 3.1416;
}

static void escribir_hoja1(DatosFourier *datos) {
    FILE *fp = fopen(CSV_HOJA1, "w");
    if (!fp) { perror("Error al crear hoja1.csv"); return; }
    fprintf(fp, "%s\n\n\nx,f(x)\n", func_csv_header(datos->func_type));
    for (int i = 0; i < NUM_PUNTOS; i++)
        fprintf(fp, "%.10g,%.15g\n", datos->x_vals[i], datos->hoja1_fx[i]);
    fclose(fp);
    printf("[Padre] Archivo %s generado.\n", CSV_HOJA1);
}

static void escribir_hoja2(DatosFourier *datos) {
    int nt = datos->num_terminos;
    FILE *fp = fopen(CSV_HOJA2, "w");
    if (!fp) { perror("Error al crear hoja2.csv"); return; }
    fprintf(fp, "\n\"F(x) = Serie de Fourier - %s\"\n", func_description(datos->func_type));
    for (int n = 1; n <= nt; n++) fprintf(fp, ",n =");
    fprintf(fp, ",,,,\n");
    fprintf(fp, "x");
    for (int n = 1; n <= nt; n++) fprintf(fp, ",%d", n);
    fprintf(fp, ",a0,x,F(X),f(x)\n");
    for (int i = 0; i < NUM_PUNTOS; i++) {
        fprintf(fp, "%.10g", datos->x_vals[i]);
        for (int n = 0; n < nt; n++)
            fprintf(fp, ",%.15g", datos->hoja2_terminos[i][n]);
        fprintf(fp, ",%.15g,%.10g,%.15g,%.15g\n",
                datos->hoja2_a0, datos->x_vals[i],
                datos->hoja2_Fx[i], datos->hoja2_fx[i]);
    }
    fclose(fp);
    printf("[Padre] Archivo %s generado.\n", CSV_HOJA2);
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Uso: %s [--func TYPE] [--terms N]\n", prog);
    fprintf(stderr, "  --func TYPE   0=x^4-3x, 1=square, 2=sawtooth, 3=triangle  (def: 0)\n");
    fprintf(stderr, "  --terms N     Numero de terminos (1..%d, def: 50)\n", MAX_TERMINOS);
}

int main(int argc, char **argv) {
    int func_type = FUNC_X4;
    int num_terminos = 50;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--func") == 0 && i + 1 < argc) func_type = atoi(argv[++i]);
        else if (strcmp(argv[i], "--terms") == 0 && i + 1 < argc) num_terminos = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) { print_usage(argv[0]); return 0; }
    }
    if (func_type < 0 || func_type >= NUM_FUNC_TYPES) { fprintf(stderr, "Error: tipo invalido\n"); return 1; }
    if (num_terminos < 1 || num_terminos > MAX_TERMINOS) { fprintf(stderr, "Error: terminos invalidos\n"); return 1; }

    key_t llave_memoria;
    int shmid;
    DatosFourier *datos;
    int sem_ids[NUM_HIJOS];
    key_t llave_sem;
    pid_t pids[NUM_HIJOS];

    printf("============================================================\n");
    printf("  Serie de Fourier — Procesos (fork)\n");
    printf("  Funcion: %s  |  Terminos: %d  |  Hijos: %d\n",
           func_description(func_type), num_terminos, NUM_HIJOS);
    printf("============================================================\n\n");

    llave_memoria = ftok(ARCHIVO_LLAVE, ID_MEMORIA);
    if (llave_memoria == -1) { perror("ftok memoria"); exit(1); }
    shmid = shmget(llave_memoria, sizeof(DatosFourier), IPC_CREAT | PERMISOS);
    if (shmid == -1) { perror("shmget"); exit(1); }
    datos = (DatosFourier *)shmat(shmid, NULL, 0);
    if (datos == (void *)-1) { perror("shmat"); exit(1); }
    memset(datos, 0, sizeof(DatosFourier));
    datos->num_terminos = num_terminos;
    datos->func_type = func_type;

    generar_puntos_x(datos->x_vals);
    datos->hoja2_a0 = a0_func(func_type);

    for (int i = 0; i < NUM_HIJOS; i++) {
        llave_sem = ftok(ARCHIVO_LLAVE, ID_SEM_BASE + i);
        if (llave_sem == -1) { perror("ftok semaforo"); exit(1); }
        sem_ids[i] = Crea_semaforo(llave_sem, 0);
    }

    int filas_por_hijo = NUM_PUNTOS / NUM_HIJOS;
    int filas_restantes = NUM_PUNTOS % NUM_HIJOS;

    for (int i = 0; i < NUM_HIJOS; i++) {
        pids[i] = fork();
        if (pids[i] < 0) { perror("fork"); exit(1); }

        if (pids[i] == 0) {
            int inicio = i * filas_por_hijo;
            int fin = inicio + filas_por_hijo;
            if (i == NUM_HIJOS - 1) fin += filas_restantes;

            down(sem_ids[i]);

            printf("[Hijo %d, PID=%d] Calculando filas [%d, %d)\n", i, getpid(), inicio, fin);

            int nt = datos->num_terminos;
            int ft = datos->func_type;
            for (int fila = inicio; fila < fin; fila++) {
                double x = datos->x_vals[fila];
                datos->hoja1_fx[fila] = f_func(x, ft);
                double suma = 0.0;
                for (int n = 1; n <= nt; n++) {
                    double term = termino_fourier_func(n, x, ft);
                    datos->hoja2_terminos[fila][n - 1] = term;
                    suma += term;
                }
                datos->hoja2_Fx[fila] = (datos->hoja2_a0 / 2.0) + suma;
                datos->hoja2_fx[fila] = f_func(x, ft);
            }

            printf("[Hijo %d, PID=%d] Calculo completado.\n", i, getpid());
            shmdt(datos);
            _exit(0);
        }
    }

    printf("[Padre] Liberando semaforos...\n");
    for (int i = 0; i < NUM_HIJOS; i++) up(sem_ids[i]);

    for (int i = 0; i < NUM_HIJOS; i++) {
        int estado;
        pid_t terminado = waitpid(pids[i], &estado, 0);
        if (terminado == -1) perror("waitpid");
    }

    printf("\n[Padre] Todos los hijos finalizaron. Escribiendo CSV...\n\n");
    escribir_hoja1(datos);
    escribir_hoja2(datos);

    shmdt(datos);
    shmctl(shmid, IPC_RMID, NULL);
    for (int i = 0; i < NUM_HIJOS; i++) semctl(sem_ids[i], 0, IPC_RMID);

    printf("\n[Padre] Recursos liberados. Programa finalizado.\n");
    return 0;
}
