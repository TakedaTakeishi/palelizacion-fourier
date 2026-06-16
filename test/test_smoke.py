import subprocess, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(cmd, cwd, desc):
    print(f"\n{desc}...", end=" ")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode == 0:
        print("OK")
        return True
    print(f"FAIL ({r.stderr[:100]})")
    return False

ok = 0
total = 4

# Sequential (Windows)
if run([os.path.join(BASE, "1_Secuencial", "fourier_seq.exe"), "--func", "0", "--terms", "10"],
       os.path.join(BASE, "1_Secuencial"), "1_Secuencial x4"):
    ok += 1
if run([os.path.join(BASE, "1_Secuencial", "fourier_seq.exe"), "--func", "1", "--terms", "5"],
       os.path.join(BASE, "1_Secuencial"), "1_Secuencial square"):
    ok += 1

# Hilos (Windows)
if run([os.path.join(BASE, "4_Hilos", "hilos_fourier.exe"), "--func", "0", "--terms", "10"],
       os.path.join(BASE, "4_Hilos"), "4_Hilos x4"):
    ok += 1
if run([os.path.join(BASE, "4_Hilos", "hilos_fourier.exe"), "--func", "1", "--terms", "5"],
       os.path.join(BASE, "4_Hilos"), "4_Hilos square"):
    ok += 1

# Procesos (WSL)
wsl_process = ["wsl", "-e", "bash", "-c",
    "cd '/mnt/c/Users/Joni/Documents/Programas/Programas Fourier/3_Procesos' && ./fourier --func 0 --terms 10"]
if run(wsl_process, BASE, "3_Procesos x4 (WSL)"):
    ok += 1
total += 1

wsl_process = ["wsl", "-e", "bash", "-c",
    "cd '/mnt/c/Users/Joni/Documents/Programas/Programas Fourier/3_Procesos' && ./fourier --func 1 --terms 5"]
if run(wsl_process, BASE, "3_Procesos square (WSL)"):
    ok += 1
total += 1

# MPI (WSL)
wsl_mpi = ["wsl", "-e", "bash", "-c",
    "cd '/mnt/c/Users/Joni/Documents/Programas/Programas Fourier/5_MPI' && mpirun --oversubscribe -np 4 ./fourier_mpi --func 0 --terms 8"]
if run(wsl_mpi, BASE, "5_MPI x4 (WSL)"):
    ok += 1
total += 1

wsl_mpi = ["wsl", "-e", "bash", "-c",
    "cd '/mnt/c/Users/Joni/Documents/Programas/Programas Fourier/5_MPI' && mpirun --oversubscribe -np 4 ./fourier_mpi --func 1 --terms 8"]
if run(wsl_mpi, BASE, "5_MPI square (WSL)"):
    ok += 1
total += 1

print(f"\n{ok}/{total} tests pasados")
sys.exit(0 if ok == total else 1)
