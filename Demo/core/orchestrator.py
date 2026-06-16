import subprocess
import time
import os
import sys
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from .types import (
    ComputationResult, METHOD_SEQUENTIAL, METHOD_THREADS,
    METHOD_PROCESSES, METHOD_MPI, FUNC_X4,
    function_real, fourier_approximation, fourier_terms_individual,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def detect_wsl() -> bool:
    try:
        r = subprocess.run(["wsl", "-e", "echo", "ok"], capture_output=True, text=True)
        return r.returncode == 0
    except FileNotFoundError:
        return False

_HAS_WSL = detect_wsl()

class ComputationMethod(ABC):
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def key(self) -> str:
        pass

    @abstractmethod
    def run(self, func_type: int, num_terms: int) -> ComputationResult:
        pass

    @abstractmethod
    def source_file(self) -> str:
        pass

    @abstractmethod
    def function_to_display(self) -> str:
        pass

class SequentialMethod(ComputationMethod):
    def name(self) -> str:
        return "Secuencial"
    def key(self) -> str:
        return METHOD_SEQUENTIAL
    def source_file(self) -> str:
        return os.path.join(BASE_DIR, "1_Secuencial", "fourier_seq.c")
    def function_to_display(self) -> str:
        return "main"

    def run(self, func_type: int, num_terms: int) -> ComputationResult:
        exe = os.path.join(BASE_DIR, "1_Secuencial", "fourier_seq.exe")
        if not os.path.exists(exe):
            exe = os.path.join(BASE_DIR, "1_Secuencial", "fourier_seq")
        cwd = os.path.join(BASE_DIR, "1_Secuencial")
        return _run_c_program(exe, cwd, self.key(), func_type, num_terms)

class ThreadsMethod(ComputationMethod):
    def name(self) -> str:
        return "Hilos (pthreads)"
    def key(self) -> str:
        return METHOD_THREADS
    def source_file(self) -> str:
        return os.path.join(BASE_DIR, "4_Hilos", "hilos_fourier.c")
    def function_to_display(self) -> str:
        return "rutina_hilo"

    def run(self, func_type: int, num_terms: int) -> ComputationResult:
        exe = os.path.join(BASE_DIR, "4_Hilos", "hilos_fourier.exe")
        if not os.path.exists(exe):
            exe = os.path.join(BASE_DIR, "4_Hilos", "hilos_fourier")
        cwd = os.path.join(BASE_DIR, "4_Hilos")
        return _run_c_program(exe, cwd, self.key(), func_type, num_terms)

class WSLMethod(ComputationMethod):
    def __init__(self, name, key, rel_dir, exe_name, source_rel, func_name, mpi=False):
        self._name = name
        self._key = key
        self._rel_dir = rel_dir
        self._exe_name = exe_name
        self._source_rel = source_rel
        self._func_name = func_name
        self._mpi = mpi

    def name(self) -> str:
        return self._name
    def key(self) -> str:
        return self._key
    def source_file(self) -> str:
        return os.path.join(BASE_DIR, self._source_rel)
    def function_to_display(self) -> str:
        return self._func_name

    def run(self, func_type: int, num_terms: int) -> ComputationResult:
        win_dir = os.path.join(BASE_DIR, self._rel_dir)
        wsl_path = win_dir.replace("C:\\", "/mnt/c/").replace("\\", "/")
        if self._mpi:
            cmd = f"cd '{wsl_path}' && mpirun --oversubscribe -np 4 ./{self._exe_name} --func {func_type} --terms {num_terms}"
        else:
            cmd = f"cd '{wsl_path}' && ./{self._exe_name} --func {func_type} --terms {num_terms}"
        full_cmd = ["wsl", "-e", "bash", "-c", cmd]

        t0 = time.perf_counter()
        result = subprocess.run(full_cmd, capture_output=True, text=True)
        t1 = time.perf_counter()

        if result.returncode != 0:
            raise RuntimeError(f"Error en {self._name}: {result.stderr}")

        csv_path = os.path.join(win_dir, "hoja2.csv")
        return _parse_csv(csv_path, self._key, func_type, num_terms, t1 - t0)

def _run_c_program(exe: str, cwd: str, method: str, func_type: int, num_terms: int) -> ComputationResult:
    cmd = [exe, "--func", str(func_type), "--terms", str(num_terms)]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    t1 = time.perf_counter()
    if result.returncode != 0:
        raise RuntimeError(f"Error en {method}: {result.stderr}")
    csv_path = os.path.join(cwd, "hoja2.csv")
    return _parse_csv(csv_path, method, func_type, num_terms, t1 - t0)

def _parse_csv(csv_path: str, method: str, func_type: int, num_terms: int, elapsed: float) -> ComputationResult:
    df = pd.read_csv(csv_path, skiprows=3)
    x = df["x"].values.astype(np.float64)
    approx = df["F(X)"].values.astype(np.float64)
    f_real = df["f(x)"].values.astype(np.float64)

    terms_arr = np.zeros((num_terms, len(x)))
    for n in range(1, num_terms + 1):
        col = str(n)
        if col in df.columns:
            terms_arr[n - 1] = df[col].values.astype(np.float64)

    return ComputationResult(
        method=method,
        func_type=func_type,
        num_terms=num_terms,
        elapsed=elapsed,
        x=x,
        f_real=f_real,
        fourier_approx=approx,
        terms=terms_arr,
        csv_path=csv_path,
    )

class Orchestrator:
    def __init__(self):
        self.methods: list[ComputationMethod] = []

        self.methods.append(SequentialMethod())
        self.methods.append(ThreadsMethod())

        if _HAS_WSL:
            self.methods.append(WSLMethod(
                "Procesos (fork)", METHOD_PROCESSES,
                "3_Procesos", "fourier", "3_Procesos/fourier.c",
                "main"
            ))
            self.methods.append(WSLMethod(
                "MPI", METHOD_MPI,
                "5_MPI", "fourier_mpi", "5_MPI/fourier_mpi.c",
                "main", mpi=True
            ))

    def disponible(self) -> list[ComputationMethod]:
        return self.methods

    def correr_todos(self, func_type: int, num_terms: int) -> list[ComputationResult]:
        results = []
        for m in self.methods:
            print(f"  Ejecutando {m.name()}…")
            r = m.run(func_type, num_terms)
            results.append(r)
            print(f"    {r.elapsed:.4f}s")
        return results

    def precomputar_standby(self, func_type: int, num_terms: int, num_puntos: int = 500) -> dict:
        x = np.linspace(-np.pi, np.pi, num_puntos)
        f_real = function_real(x, func_type)
        terms = fourier_terms_individual(x, num_terms, func_type)
        approx = fourier_approximation(x, num_terms, func_type)
        return {
            "x": x,
            "f_real": f_real,
            "terms": terms,
            "approx": approx,
            "num_terms": num_terms,
            "func_type": func_type,
        }
