#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../common/fourier_core.h"

#define CSV_HOJA1 "hoja1.csv"
#define CSV_HOJA2 "hoja2.csv"

static void generar_puntos_x(double *x_vals) {
    x_vals[0] = -3.1416;
    for (int i = 1; i <= 62; i++) {
        x_vals[i] = -3.1416 + i * 0.1;
    }
    x_vals[63] = 3.1416;
}

static void escribir_hoja1(DatosFourier *datos) {
    FILE *fp = fopen(CSV_HOJA1, "w");
    if (!fp) { perror("Error al crear hoja1.csv"); return; }
    fprintf(fp, "%s\n\n\nx,f(x)\n", func_csv_header(datos->func_type));
    for (int i = 0; i < NUM_PUNTOS; i++) {
        fprintf(fp, "%.10g,%.15g\n", datos->x_vals[i], datos->hoja1_fx[i]);
    }
    fclose(fp);
    printf("Archivo %s generado.\n", CSV_HOJA1);
}

static void escribir_hoja2(DatosFourier *datos) {
    FILE *fp = fopen(CSV_HOJA2, "w");
    if (!fp) { perror("Error al crear hoja2.csv"); return; }
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

static void print_usage(const char *prog) {
    fprintf(stderr, "Uso: %s [--func TYPE] [--terms N]\n", prog);
    fprintf(stderr, "  --func TYPE   0=x^4-3x, 1=square, 2=sawtooth, 3=triangle  (def: 0)\n");
    fprintf(stderr, "  --terms N     Numero de terminos (1..%lld, def: 50)\n", MAX_TERMINOS);
}

int main(int argc, char **argv) {
    int func_type = FUNC_X4;
    long long num_terminos = 50;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--func") == 0 && i + 1 < argc) {
            func_type = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--terms") == 0 && i + 1 < argc) {
            num_terminos = atoll(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }
    if (func_type < 0 || func_type >= NUM_FUNC_TYPES) {
        fprintf(stderr, "Error: tipo de funcion invalido: %d\n", func_type);
        return 1;
    }
    if (num_terminos < 1 || num_terminos > MAX_TERMINOS) {
        fprintf(stderr, "Error: numero de terminos debe estar entre 1 y %lld\n", MAX_TERMINOS);
        return 1;
    }

    DatosFourier *datos = (DatosFourier *)malloc(sizeof(DatosFourier));
    if (!datos) { perror("Error allocating memory"); return 1; }
    memset(datos, 0, sizeof(DatosFourier));
    datos->num_terminos = num_terminos;
    datos->func_type = func_type;

    printf("============================================================\n");
    printf("  Serie de Fourier — Secuencial\n");
    printf("  Funcion: %s  |  Terminos: %lld\n", func_description(func_type), num_terminos);
    printf("============================================================\n\n");

    generar_puntos_x(datos->x_vals);
    datos->hoja2_a0 = a0_func(func_type);

    for (int fila = 0; fila < NUM_PUNTOS; fila++) {
        double x = datos->x_vals[fila];
        datos->hoja1_fx[fila] = f_func(x, func_type);
        double suma = 0.0;
        for (long long n = 1; n <= num_terminos; n++) {
            double term = termino_fourier_func(n, x, func_type);
            if (n <= MAX_SAVED_TERMINOS) {
                datos->hoja2_terminos[fila][n - 1] = term;
            }
            suma += term;
        }
        datos->hoja2_Fx[fila] = (datos->hoja2_a0 / 2.0) + suma;
        datos->hoja2_fx[fila] = f_func(x, func_type);
    }

    printf("Calculo completado. Escribiendo archivos...\n\n");
    escribir_hoja1(datos);
    escribir_hoja2(datos);
    free(datos);

    printf("\nPrograma finalizado con exito.\n");
    return 0;
}
