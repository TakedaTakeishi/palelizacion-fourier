import subprocess
import os
import sys
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = [
    ("Secuencial", BASE, ["1_Secuencial/fourier_seq"]),
    ("Hilos",      BASE, ["4_Hilos/hilos_fourier"]),
]

WSL_METHODS = [
    ("Procesos", ["wsl", "-e", "bash", "-c",
                  "cd '/mnt/c/Users/Joni/Documents/Programas/Programas Fourier/3_Procesos' && ./fourier"]),
    ("MPI",      ["wsl", "-e", "bash", "-c",
                  "cd '/mnt/c/Users/Joni/Documents/Programas/Programas Fourier/5_MPI' && mpirun --oversubscribe -np 4 ./fourier_mpi"]),
]

def run_method(cmd, cwd, func, terms, skiprows=3):
    full_cmd = cmd + [f"--func", str(func), "--terms", str(terms)]
    result = subprocess.run(full_cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None
    csv_path = os.path.join(cwd if cwd else "", "hoja2.csv")
    if not os.path.exists(csv_path):
        print(f"  ERROR: no se genero hoja2.csv en {cwd}")
        return None
    try:
        df = pd.read_csv(csv_path, skiprows=skiprows)
        return df
    except Exception as e:
        print(f"  ERROR leyendo CSV: {e}")
        return None

def test_func(name, cmd, cwd, func_id, terms, skiprows):
    print(f"\n{'='*50}")
    print(f"  {name} | func={func_id} terms={terms}")
    print(f"{'='*50}")
    df = run_method(cmd + [""] if not cmd[0].endswith('.exe') else cmd, cwd, func_id, terms, skiprows)
    if df is None:
        return False
    print(f"  Columnas: {list(df.columns)}")
    print(f"  Filas: {len(df)}")
    print(f"  Primer F(X): {df['F(X)'].iloc[0]:.6f}")
    print(f"  Ultimo F(X):  {df['F(X)'].iloc[-1]:.6f}")
    return True

def main():
    ok = 0
    total = 0
    
    for name, cwd, cmd in METHODS:
        for func_id in [0, 1]:
            total += 1
            if test_func(name, cmd, cwd, func_id, 10, 3):
                ok += 1

    for name, cmd in WSL_METHODS:
        for func_id in [0, 1]:
            total += 1
            print(f"\n{'='*50}")
            print(f"  {name} (WSL) | func={func_id} terms=10")
            print(f"{'='*50}")
            full_cmd = cmd + [f"--func", str(func_id), "--terms", "10"]
            result = subprocess.run(full_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr}")
                continue
            csv_path = os.path.join(BASE, name == "Procesos" and "3_Procesos" or "5_MPI", "hoja2.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, skiprows=3)
                print(f"  Columnas: {list(df.columns)}")
                print(f"  Filas: {len(df)}")
                print(f"  Primer F(X): {df['F(X)'].iloc[0]:.6f}")
                print(f"  Ultimo F(X):  {df['F(X)'].iloc[-1]:.6f}")
                ok += 1
            else:
                print(f"  ERROR: CSV no encontrado en {csv_path}")
                total -= 1
    
    print(f"\n{'='*50}")
    print(f"  {ok}/{total} pruebas pasadas")
    if ok < total:
        sys.exit(1)

if __name__ == "__main__":
    main()
