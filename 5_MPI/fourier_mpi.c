#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../common/fourier_core.h"

#define CSV_HOJA1 "hoja1.csv"
#define CSV_HOJA2 "hoja2.csv"

static void escribir_hoja1(DatosFourier *datos) {
    FILE *fp = fopen(CSV_HOJA1, "w");
    if (!fp) { perror("Error al crear hoja1.csv"); return; }
    fprintf(fp, "%s\n\n\nx,f(x)\n", func_csv_header(datos->func_type));
    for (int i = 0; i < NUM_PUNTOS; i++) {
        fprintf(fp, "%.10g,%.15g\n", datos->x_vals[i], datos->hoja1_fx[i]);
    }
    fclose(fp);
    printf("[Maestro] Archivo %s generado.\n", CSV_HOJA1);
}

static void escribir_hoja2(DatosFourier *datos) {
    long long nt = datos->num_terminos;
    FILE *fp = fopen(CSV_HOJA2, "w");
    if (!fp) { perror("Error al crear hoja2.csv"); return; }
    fprintf(fp, "\n\"F(x) = Serie de Fourier - %s\"\n", func_description(datos->func_type));
    long long cols = (nt < MAX_SAVED_TERMINOS) ? nt : MAX_SAVED_TERMINOS;
    for (long long n = 1; n <= cols; n++) fprintf(fp, ",n =");
    fprintf(fp, ",,,,\n");
    fprintf(fp, "x");
    for (long long n = 1; n <= cols; n++) fprintf(fp, ",%lld", n);
    fprintf(fp, ",a0,x,F(X),f(x)\n");
    for (int i = 0; i < NUM_PUNTOS; i++) {
        fprintf(fp, "%.10g", datos->x_vals[i]);
        for (long long n = 0; n < cols; n++)
            fprintf(fp, ",%.15g", datos->hoja2_terminos[i][n]);
        fprintf(fp, ",%.15g,%.10g,%.15g,%.15g\n",
                datos->hoja2_a0, datos->x_vals[i],
                datos->hoja2_Fx[i], datos->hoja2_fx[i]);
    }
    fclose(fp);
    printf("[Maestro] Archivo %s generado.\n", CSV_HOJA2);
}

static void generar_puntos_x(double *x_vals) {
    x_vals[0] = -3.1416;
    for (int i = 1; i <= 62; i++)
        x_vals[i] = -3.1416 + i * 0.1;
    x_vals[63] = 3.1416;
}

int main(int argc, char **argv) {
    int rank = 0, world_size = 0;
    int func_type = FUNC_X4;
    long long num_terminos = 50;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0) {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--func") == 0 && i + 1 < argc) func_type = atoi(argv[++i]);
            else if (strcmp(argv[i], "--terms") == 0 && i + 1 < argc) num_terminos = atoll(argv[++i]);
        }
        if (func_type < 0 || func_type >= NUM_FUNC_TYPES) {
            fprintf(stderr, "Error: tipo de funcion invalido\n"); MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (num_terminos < 1 || num_terminos > MAX_TERMINOS) {
            fprintf(stderr, "Error: terminos invalidos\n"); MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&func_type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_terminos, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (world_size != 4) {
        if (rank == 0) printf("Error: se requieren exactamente 4 procesos.\n");
        MPI_Finalize(); return 0;
    }

    DatosFourier *datos = (DatosFourier *)malloc(sizeof(DatosFourier));
    if (!datos) { perror("malloc datos"); MPI_Abort(MPI_COMM_WORLD, 1); }
    memset(datos, 0, sizeof(DatosFourier));
    datos->num_terminos = num_terminos;
    datos->func_type = func_type;

    if (rank == 0) {
        printf("============================================================\n");
        printf("  Serie de Fourier — MPI\n");
        printf("  Funcion: %s  |  Terminos: %lld  |  Procesos: %d\n",
               func_description(func_type), num_terminos, world_size);
        printf("============================================================\n\n");
        generar_puntos_x(datos->x_vals);
        datos->hoja2_a0 = a0_func(func_type);
    }

    MPI_Bcast(datos->x_vals, NUM_PUNTOS, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&datos->hoja2_a0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int workers = world_size - 1;
    int local_start = 0, local_end = 0;

    if (rank > 0) {
        int worker_index = rank - 1;
        int rows_base = NUM_PUNTOS / workers;
        int rows_rem = NUM_PUNTOS % workers;
        int extra = (worker_index < rows_rem) ? 1 : 0;
        local_start = worker_index * rows_base + (worker_index < rows_rem ? worker_index : rows_rem);
        local_end = local_start + rows_base + extra;
    }

    int local_count = (local_end > local_start) ? (local_end - local_start) : 0;
    long long saved_terms = (num_terminos < MAX_SAVED_TERMINOS) ? num_terminos : MAX_SAVED_TERMINOS;

    double *local_hoja1 = local_count > 0 ? (double *)calloc(local_count, sizeof(double)) : NULL;
    double *local_terminos = local_count > 0 ? (double *)calloc(local_count * saved_terms, sizeof(double)) : NULL;
    double *local_Fx = local_count > 0 ? (double *)calloc(local_count, sizeof(double)) : NULL;
    double *local_fx = local_count > 0 ? (double *)calloc(local_count, sizeof(double)) : NULL;

    if (rank > 0 && local_count > 0) {
        long long nt = num_terminos;
        int ft = func_type;
        printf("[Trabajador %d] Calculando filas [%d, %d)\n", rank, local_start, local_end);
        for (int fila = local_start; fila < local_end; fila++) {
            int idx = fila - local_start;
            double x = datos->x_vals[fila];
            local_hoja1[idx] = f_func(x, ft);
            double suma = 0.0;
            for (long long n = 1; n <= nt; n++) {
                double term = termino_fourier_func(n, x, ft);
                if (n <= saved_terms) {
                    local_terminos[idx * saved_terms + (n - 1)] = term;
                }
                suma += term;
            }
            local_Fx[idx] = (datos->hoja2_a0 / 2.0) + suma;
            local_fx[idx] = f_func(x, ft);
        }
        printf("[Trabajador %d] Trabajo finalizado.\n", rank);
    }

    int recv_counts[4] = {0,0,0,0};
    int recv_displs[4] = {0,0,0,0};

    if (rank == 0) {
        int offset = 0;
        for (int r = 1; r < world_size; r++) {
            int wi = r - 1;
            int rb = NUM_PUNTOS / workers;
            int rr = NUM_PUNTOS % workers;
            int extra = (wi < rr) ? 1 : 0;
            int start = wi * rb + (wi < rr ? wi : rr);
            int end = start + rb + extra;
            int count = (end > start) ? (end - start) : 0;
            recv_counts[r] = count;
            recv_displs[r] = offset;
            offset += count;
        }
    }

    MPI_Gatherv(local_hoja1, local_count, MPI_DOUBLE,
                datos->hoja1_fx, recv_counts, recv_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_Fx, local_count, MPI_DOUBLE,
                datos->hoja2_Fx, recv_counts, recv_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_fx, local_count, MPI_DOUBLE,
                datos->hoja2_fx, recv_counts, recv_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int recv_counts_terms[4] = {0, 0, 0, 0};
        int recv_displs_terms[4] = {0, 0, 0, 0};
        for (int r = 1; r < world_size; r++) {
            int wi = r - 1;
            int rb = NUM_PUNTOS / workers;
            int rr = NUM_PUNTOS % workers;
            int extra = (wi < rr) ? 1 : 0;
            int start = wi * rb + (wi < rr ? wi : rr);
            int end = start + rb + extra;
            int count = (end > start) ? (end - start) : 0;
            recv_counts_terms[r] = count * saved_terms;
            recv_displs_terms[r] = (r == 1) ? 0 : recv_displs_terms[r-1] + recv_counts_terms[r-1];
        }
        double *flat_terms = calloc(NUM_PUNTOS * saved_terms, sizeof(double));
        if (flat_terms) {
            MPI_Gatherv(local_terminos, local_count * saved_terms, MPI_DOUBLE,
                        flat_terms, recv_counts_terms, recv_displs_terms,
                        MPI_DOUBLE, 0, MPI_COMM_WORLD);
            for (int i = 0; i < NUM_PUNTOS; i++) {
                for (int n = 0; n < saved_terms; n++) {
                    datos->hoja2_terminos[i][n] = flat_terms[i * saved_terms + n];
                }
            }
            free(flat_terms);
        }
    } else {
        int zero_counts[4] = {0, 0, 0, 0};
        int zero_displs[4] = {0, 0, 0, 0};
        MPI_Gatherv(local_terminos, local_count * saved_terms, MPI_DOUBLE,
                    NULL, zero_counts, zero_displs,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("\n[Maestro] Todos los trabajadores terminaron. Escribiendo CSV...\n\n");
        escribir_hoja1(datos);
        escribir_hoja2(datos);
        printf("\n[Maestro] Programa finalizado.\n");
    }

    free(local_hoja1); free(local_terminos); free(local_Fx); free(local_fx);
    free(datos);
    MPI_Finalize();
    return 0;
}
