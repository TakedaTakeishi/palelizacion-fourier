import subprocess
import sys
import os
import platform

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(cmd, cwd, name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Comando: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr}")
        return False
    print(f"  OK")
    if result.stderr:
        for l in result.stderr.strip().split('\n'):
            if 'warning' in l.lower() or 'warn' in l.lower():
                print(f"  WARN: {l.strip()}")
    return True

def main():
    results = []
    
    results.append(run(
        ["gcc", "-Wall", "-O2", "-o", "fourier_seq", "fourier_seq.c", "-lm"],
        os.path.join(BASE, "1_Secuencial"),
        "1_Secuencial (Windows)"
    ))
    
    results.append(run(
        ["gcc", "-Wall", "-O2", "-o", "hilos_fourier", "hilos_fourier.c", "-lm", "-lpthread"],
        os.path.join(BASE, "4_Hilos"),
        "4_Hilos (Windows)"
    ))
    
    if platform.system() == "Windows":
        print("\n  Nota: procesos y MPI requieren WSL/Linux para compilar")
        is_wsl = subprocess.run(["wsl", "-e", "echo", "ok"], capture_output=True).returncode == 0
        if is_wsl:
            results.append(run(
                ["wsl", "-e", "bash", "-c",
                 "cd '{}' && gcc -Wall -O2 -o fourier fourier.c -lm".format(
                     os.path.join(BASE, "3_Procesos").replace('\\', '/').replace('C:', '/mnt/c'))],
                BASE,
                "3_Procesos (WSL)"
            ))
            results.append(run(
                ["wsl", "-e", "bash", "-c",
                 "cd '{}' && mpicc -Wall -O2 -o fourier_mpi fourier_mpi.c -lm".format(
                     os.path.join(BASE, "5_MPI").replace('\\', '/').replace('C:', '/mnt/c'))],
                BASE,
                "5_MPI (WSL)"
            ))
        else:
            print("  WSL no disponible, saltando procesos/MPI")
    else:
        results.append(run(["gcc", "-Wall", "-O2", "-o", "fourier", "fourier.c", "-lm"],
                           os.path.join(BASE, "3_Procesos"), "3_Procesos"))
        results.append(run(["mpicc", "-Wall", "-O2", "-o", "fourier_mpi", "fourier_mpi.c", "-lm"],
                           os.path.join(BASE, "5_MPI"), "5_MPI"))

    ok = sum(1 for r in results if r)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"  {ok}/{total} compilaciones exitosas")
    if ok < total:
        sys.exit(1)

if __name__ == "__main__":
    main()
