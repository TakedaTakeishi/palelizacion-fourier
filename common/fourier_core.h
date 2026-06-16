#ifndef FOURIER_CORE_H
#define FOURIER_CORE_H

#include <math.h>
#include <stdio.h>

#define MAX_TERMINOS   10000
#define NUM_PUNTOS      64

#define FUNC_X4         0
#define FUNC_SQUARE     1
#define FUNC_SAWTOOTH   2
#define FUNC_TRIANGLE   3
#define NUM_FUNC_TYPES  4

static const char *FUNC_DESCRIPTIONS[] = {
    "f(x) = x^4 - 3x",
    "Square wave",
    "Sawtooth wave",
    "Triangle wave"
};

static const char *FUNC_CSV_HEADERS[] = {
    "\"f(x) = (x^4) - 3x\"",
    "\"f(x) = Square wave\"",
    "\"f(x) = Sawtooth wave\"",
    "\"f(x) = Triangle wave\""
};

typedef struct {
    double x_vals[NUM_PUNTOS];
    double hoja1_fx[NUM_PUNTOS];
    double hoja2_terminos[NUM_PUNTOS][MAX_TERMINOS];
    double hoja2_a0;
    double hoja2_Fx[NUM_PUNTOS];
    double hoja2_fx[NUM_PUNTOS];
    int    num_terminos;
    int    func_type;
} DatosFourier;

static inline double f_func(double x, int func_type) {
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

static inline double termino_fourier_func(int n, double x, int func_type) {
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

static inline double a0_func(int func_type) {
    switch (func_type) {
        case FUNC_X4:
            return (2.0 * pow(M_PI, 4)) / 5.0;
        default:
            return 0.0;
    }
}

static inline const char* func_description(int func_type) {
    if (func_type >= 0 && func_type < NUM_FUNC_TYPES)
        return FUNC_DESCRIPTIONS[func_type];
    return "Unknown function";
}

static inline const char* func_csv_header(int func_type) {
    if (func_type >= 0 && func_type < NUM_FUNC_TYPES)
        return FUNC_CSV_HEADERS[func_type];
    return "\"Unknown function\"";
}

static inline double normalize_angle(double x) {
    x = fmod(x + M_PI, 2.0 * M_PI);
    if (x < 0) x += 2.0 * M_PI;
    return x - M_PI;
}

#endif
