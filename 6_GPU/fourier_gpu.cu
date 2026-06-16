#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define MAX_TERMINOS   10000
#define NUM_PUNTOS      64

#define FUNC_X4         0
#define FUNC_SQUARE     1
#define FUNC_SAWTOOTH   2
#define FUNC_TRIANGLE   3
#define NUM_FUNC_TYPES  4

static const char *FUNC_CSV_HEADERS[] = {
    "\"f(x) = (x^4) - 3x\"",
    "\"f(x) = Square wave\"",
    "\"f(x) = Sawtooth wave\"",
    "\"f(x) = Triangle wave\""
};

static const char *FUNC_DESCRIPTIONS[] = {
    "f(x) = x^4 - 3x",
    "Square wave",
    "Sawtooth wave",
    "Triangle wave"
};

__device__ double dev_f_func(double x, int func_type) {
    switch (func_type) {
        case FUNC_X4:
            return pow(x, 4) - 3.0 * x;
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

__device__ double dev_termino_fourier(int n, double x, int func_type) {
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
            int k = (n - 1) / 2;
            double signo = (k % 2 == 0) ? 1.0 : -1.0;
            return (8.0 * signo / (M_PI * M_PI * (double)n * (double)n)) * sin((double)n * x);
        }
        default:
            return 0.0;
    }
}

__host__ __device__ double dev_a0_func(int func_type) {
    switch (func_type) {
        case FUNC_X4:
            return (2.0 * pow(M_PI, 4)) / 5.0;
        default:
            return 0.0;
    }
}

__global__ void compute_fourier_kernel(
    double *x_vals,
    double *f_real,
    double *fourier_approx,
    double *terms,
    double a0_half,
    int num_terminos,
    int func_type,
    int n_puntos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_puntos) return;

    double x = x_vals[idx];
    f_real[idx] = dev_f_func(x, func_type);

    double suma = 0.0;
    for (int n = 1; n <= num_terminos; n++) {
        double term = dev_termino_fourier(n, x, func_type);
        terms[idx * num_terminos + (n - 1)] = term;
        suma += term;
    }
    fourier_approx[idx] = a0_half + suma;
}

static void generar_puntos_x(double *x_vals) {
    x_vals[0] = -3.1416;
    for (int i = 1; i <= 62; i++) {
        x_vals[i] = -3.1416 + i * 0.1;
    }
    x_vals[63] = 3.1416;
}

static void escribir_csv(
    const char *csv_path,
    double *x_vals,
    double *f_real,
    double *fourier_approx,
    double *terms,
    double a0_half,
    int num_terminos,
    int func_type
) {
    FILE *fp = fopen(csv_path, "w");
    if (!fp) { perror("Error al crear CSV"); return; }

    fprintf(fp, "\n\"F(x) = Serie de Fourier - %s\"\n", FUNC_DESCRIPTIONS[func_type]);
    for (int n = 1; n <= num_terminos; n++) fprintf(fp, ",n =");
    fprintf(fp, ",,,,\n");
    fprintf(fp, "x");
    for (int n = 1; n <= num_terminos; n++) fprintf(fp, ",%d", n);
    fprintf(fp, ",a0,x,F(X),f(x)\n");

    for (int i = 0; i < NUM_PUNTOS; i++) {
        fprintf(fp, "%.10g", x_vals[i]);
        for (int n = 0; n < num_terminos; n++) {
            fprintf(fp, ",%.15g", terms[i * num_terminos + n]);
        }
        fprintf(fp, ",%.15g,%.10g,%.15g,%.15g\n",
                a0_half * 2.0, x_vals[i],
                fourier_approx[i], f_real[i]);
    }
    fclose(fp);
    printf("Archivo %s generado.\n", csv_path);
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
        if (strcmp(argv[i], "--func") == 0 && i + 1 < argc) {
            func_type = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--terms") == 0 && i + 1 < argc) {
            num_terminos = atoi(argv[++i]);
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
        fprintf(stderr, "Error: numero de terminos debe estar entre 1 y %d\n", MAX_TERMINOS);
        return 1;
    }

    printf("============================================================\n");
    printf("  Serie de Fourier — GPU (CUDA)\n");
    printf("  Funcion: %s  |  Terminos: %d\n", FUNC_DESCRIPTIONS[func_type], num_terminos);
    printf("============================================================\n\n");

    double h_x[NUM_PUNTOS];
    double h_f_real[NUM_PUNTOS];
    double h_fourier_approx[NUM_PUNTOS];
    double *h_terms = (double *)malloc(NUM_PUNTOS * num_terminos * sizeof(double));
    if (!h_terms) { perror("Error allocating host memory"); return 1; }

    generar_puntos_x(h_x);

    double a0_val = dev_a0_func(func_type);
    double a0_half = a0_val / 2.0;

    double *d_x, *d_f_real, *d_fourier_approx, *d_terms;
    size_t terms_size = NUM_PUNTOS * num_terminos * sizeof(double);

    cudaMalloc(&d_x, NUM_PUNTOS * sizeof(double));
    cudaMalloc(&d_f_real, NUM_PUNTOS * sizeof(double));
    cudaMalloc(&d_fourier_approx, NUM_PUNTOS * sizeof(double));
    cudaMalloc(&d_terms, terms_size);

    cudaMemcpy(d_x, h_x, NUM_PUNTOS * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 64;
    int blocks = (NUM_PUNTOS + threads - 1) / threads;

    compute_fourier_kernel<<<blocks, threads>>>(
        d_x, d_f_real, d_fourier_approx, d_terms,
        a0_half, num_terminos, func_type, NUM_PUNTOS
    );

    cudaDeviceSynchronize();

    cudaMemcpy(h_f_real, d_f_real, NUM_PUNTOS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fourier_approx, d_fourier_approx, NUM_PUNTOS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_terms, d_terms, terms_size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_f_real);
    cudaFree(d_fourier_approx);
    cudaFree(d_terms);

    printf("Calculo completado. Escribiendo archivos...\n\n");
    escribir_csv("hoja2.csv", h_x, h_f_real, h_fourier_approx, h_terms,
                 a0_half, num_terminos, func_type);

    free(h_terms);

    printf("\nPrograma finalizado con exito.\n");
    return 0;
}
