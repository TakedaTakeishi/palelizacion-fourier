from dataclasses import dataclass, field
from typing import Optional
import numpy as np

FUNC_X4 = 0
FUNC_SQUARE = 1
FUNC_SAWTOOTH = 2
FUNC_TRIANGLE = 3

FUNC_NAMES = {
    FUNC_X4: "f(x) = x⁴ − 3x",
    FUNC_SQUARE: "Square wave",
    FUNC_SAWTOOTH: "Sawtooth wave",
    FUNC_TRIANGLE: "Triangle wave",
}

FUNC_SHORT_NAMES = {
    FUNC_X4: "x⁴−3x",
    FUNC_SQUARE: "Square",
    FUNC_SAWTOOTH: "Sawtooth",
    FUNC_TRIANGLE: "Triangle",
}

METHOD_SEQUENTIAL = "secuencial"
METHOD_THREADS = "hilos"
METHOD_PROCESSES = "procesos"
METHOD_MPI = "mpi"
METHOD_GPU = "gpu"

METHOD_NAMES = {
    METHOD_SEQUENTIAL: "Secuencial",
    METHOD_THREADS: "Hilos (pthreads)",
    METHOD_PROCESSES: "Procesos (fork)",
    METHOD_MPI: "MPI",
    METHOD_GPU: "GPU (CUDA)",
}

METHOD_COLORS = {
    METHOD_SEQUENTIAL: (102, 252, 241),
    METHOD_THREADS: (255, 0, 127),
    METHOD_PROCESSES: (57, 255, 20),
    METHOD_MPI: (255, 209, 102),
    METHOD_GPU: (113, 29, 154),
}

def function_real(x: np.ndarray, func_type: int) -> np.ndarray:
    if func_type == FUNC_X4:
        return x**4 - 3 * x
    elif func_type == FUNC_SQUARE:
        return np.where(x > 0, 1.0, -1.0)
    elif func_type == FUNC_SAWTOOTH:
        xn = np.mod(x + np.pi, 2 * np.pi)
        xn[xn < 0] += 2 * np.pi
        xn -= np.pi
        boundary = (xn <= -np.pi + 1e-12) | (xn >= np.pi - 1e-12)
        result = xn / np.pi
        result[boundary] = 0.0
        return result
    elif func_type == FUNC_TRIANGLE:
        xn = np.mod(x + np.pi, 2 * np.pi)
        xn[xn < 0] += 2 * np.pi
        xn -= np.pi
        boundary = (xn <= -np.pi + 1e-12) | (xn >= np.pi - 1e-12)
        result = np.zeros_like(x)
        mid = (xn >= -np.pi/2) & (xn <= np.pi/2) & ~boundary
        left = (xn < -np.pi/2) & ~boundary
        right = (xn > np.pi/2) & ~boundary
        result[mid] = 2.0 * xn[mid] / np.pi
        result[left] = -2.0 * (xn[left] + np.pi) / np.pi
        result[right] = -2.0 * (xn[right] - np.pi) / np.pi
        result[boundary] = 0.0
        return result
    return np.zeros_like(x)

def fourier_approximation(x: np.ndarray, num_terms: int, func_type: int) -> np.ndarray:
    result = np.zeros_like(x)
    a0_val = a0_func(func_type)
    result += a0_val / 2.0
    for n in range(1, num_terms + 1):
        result += termino_fourier(n, x, func_type)
    return result

def fourier_terms_individual(x: np.ndarray, num_terms: int, func_type: int) -> np.ndarray:
    terms = np.zeros((num_terms, len(x)))
    for n in range(1, num_terms + 1):
        terms[n - 1] = termino_fourier(n, x, func_type)
    return terms

def termino_fourier(n: int, x: np.ndarray, func_type: int) -> np.ndarray:
    if func_type == FUNC_X4:
        n2 = float(n * n)
        n4 = n2 * n2
        signo = 1.0 if n % 2 == 0 else -1.0
        pi2 = np.pi * np.pi
        an = (8.0 * pi2 * n2 - 48.0) * signo / n4
        bn = 6.0 * signo / float(n)
        return an * np.cos(n * x) + bn * np.sin(n * x)
    elif func_type == FUNC_SQUARE:
        if n % 2 == 0:
            return np.zeros_like(x)
        return (4.0 / (np.pi * n)) * np.sin(n * x)
    elif func_type == FUNC_SAWTOOTH:
        signo = -1.0 if n % 2 == 0 else 1.0
        return (2.0 * signo / (np.pi * n)) * np.sin(n * x)
    elif func_type == FUNC_TRIANGLE:
        if n % 2 == 0:
            return np.zeros_like(x)
        k = (n - 1) // 2
        signo = 1.0 if k % 2 == 0 else -1.0
        return (8.0 * signo / (np.pi * np.pi * n * n)) * np.sin(n * x)
    return np.zeros_like(x)

def a0_func(func_type: int) -> float:
    if func_type == FUNC_X4:
        return (2.0 * np.pi**4) / 5.0
    return 0.0

@dataclass
class ComputationResult:
    method: str
    func_type: int
    num_terms: int
    elapsed: float
    x: np.ndarray
    f_real: np.ndarray
    fourier_approx: np.ndarray
    terms: np.ndarray
    csv_path: str
