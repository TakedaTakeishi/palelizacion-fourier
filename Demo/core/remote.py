import os
import time
import sys
from dotenv import load_dotenv

try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

GPU_MODE = os.getenv("GPU_MODE", "auto")
SSH_HOST = os.getenv("SSH_HOST", "")
SSH_PORT = int(os.getenv("SSH_PORT", "22"))
SSH_USER = os.getenv("SSH_USER", "")
SSH_PASSWORD = os.getenv("SSH_PASSWORD", "")

_REMOTE_DIR = "~/fourier_gpu"
_STATUS = {"connected": False, "mode": "unknown", "error": ""}


def get_status():
    return dict(_STATUS)


def detect_local_cuda():
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def test_connection():
    if not HAS_PARAMIKO:
        _STATUS["connected"] = False
        _STATUS["error"] = "paramiko no instalado"
        return False
    if not SSH_HOST or not SSH_USER:
        _STATUS["connected"] = False
        _STATUS["error"] = "SSH_HOST o SSH_USER vacío en .env"
        return False
    try:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=10)
        client.close()
        _STATUS["connected"] = True
        _STATUS["error"] = ""
        return True
    except Exception as e:
        _STATUS["connected"] = False
        _STATUS["error"] = str(e)
        return False


def resolve_mode():
    mode = GPU_MODE.lower()
    if mode == "local":
        if detect_local_cuda():
            _STATUS["mode"] = "local"
            return "local"
        _STATUS["mode"] = "local_fail"
        _STATUS["error"] = "CUDA local no disponible"
        return "local"

    if mode == "remote":
        test_connection()
        _STATUS["mode"] = "remote"
        return "remote"

    if detect_local_cuda():
        _STATUS["mode"] = "local"
        return "local"

    if test_connection():
        _STATUS["mode"] = "remote"
        return "remote"

    _STATUS["mode"] = "unavailable"
    _STATUS["error"] = "Sin CUDA local ni conexión SSH"
    return "unavailable"


def _resolve_remote_dir(client):
    _, stdout, _ = client.exec_command("echo $HOME")
    home = stdout.read().decode().strip()
    return f"{home}/fourier_gpu"


def run_remote(func_type, num_terms):
    if not HAS_PARAMIKO:
        raise RuntimeError("paramiko no está instalado. Ejecuta: uv add paramiko")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASSWORD, timeout=15)

    remote_dir = _resolve_remote_dir(client)
    _, stdout, _ = client.exec_command(f"mkdir -p {remote_dir}")
    stdout.channel.recv_exit_status()

    sftp = client.open_sftp()
    cu_local = os.path.join(os.path.dirname(__file__), "..", "..", "7_CUDA", "fourier.cu")
    cu_remote = f"{remote_dir}/fourier.cu"
    sftp.put(cu_local, cu_remote)
    sftp.close()

    compile_cmd = f"cd {remote_dir} && nvcc -o fourier fourier.cu -lm 2>&1"
    _, stdout, stderr = client.exec_command(compile_cmd)
    exit_status = stdout.channel.recv_exit_status()
    compile_output = stdout.read().decode() + stderr.read().decode()
    if exit_status != 0:
        client.close()
        raise RuntimeError(f"Error de compilación en servidor:\n{compile_output}")

    run_cmd = f"cd {remote_dir} && ./fourier --func {func_type} --terms {num_terms} 2>&1"
    t0 = time.perf_counter()
    _, stdout, stderr = client.exec_command(run_cmd)
    exit_status = stdout.channel.recv_exit_status()
    t1 = time.perf_counter()
    run_output = stdout.read().decode() + stderr.read().decode()
    if exit_status != 0:
        client.close()
        raise RuntimeError(f"Error de ejecución en servidor:\n{run_output}")

    csv_remote = f"{remote_dir}/hoja2.csv"
    csv_local = os.path.join(os.path.dirname(__file__), "..", "..", "7_CUDA", "hoja2.csv")
    sftp2 = client.open_sftp()
    sftp2.get(csv_remote, csv_local)
    sftp2.close()

    client.exec_command(f"rm -f {remote_dir}/fourier.cu {remote_dir}/fourier {remote_dir}/hoja2.csv")

    client.close()
    return csv_local, t1 - t0
