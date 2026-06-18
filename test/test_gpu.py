import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Demo"))

from core.orchestrator import Orchestrator, METHOD_GPU
from core.remote import detect_local_cuda, test_connection, resolve_mode, get_status, GPU_MODE


def test_gpu_method_exists():
    o = Orchestrator()
    methods = [m for m in o.disponible() if m.key() == METHOD_GPU]
    assert len(methods) > 0, "GpuMethod debería estar registrado en el orquestador"
    gpu = methods[0]
    assert gpu.name() == "GPU (CUDA)"
    assert gpu.key() == METHOD_GPU
    assert gpu.source_file().endswith("fourier.cu")
    print(f"  GPU method: {gpu.name()} ✓")


def test_gpu_status():
    o = Orchestrator()
    st = o.gpu_status()
    assert "mode" in st
    assert "connected" in st
    print(f"  GPU status: mode={st['mode']}, connected={st['connected']}")


def test_resolve_mode():
    mode = resolve_mode()
    assert mode in ("local", "remote", "unavailable")
    print(f"  GPU mode resuelto: {mode}")


def test_detect_local_cuda():
    available = detect_local_cuda()
    if available:
        print("  CUDA local detectado ✓")
    else:
        print("  CUDA local no disponible (esperado en Windows sin GPU)")


def test_ssh_connection():
    if GPU_MODE != "remote":
        print("  SSH: skip (modo no es 'remote')")
        return
    connected = test_connection()
    st = get_status()
    if connected:
        print(f"  SSH conectado a {os.getenv('SSH_HOST', '?')} ✓")
    else:
        print(f"  SSH no conectado: {st.get('error', '?')}")


def test_remote_module_loads():
    from core import remote
    assert hasattr(remote, "run_remote")
    assert hasattr(remote, "resolve_mode")
    assert hasattr(remote, "test_connection")
    print("  Módulo remote.py cargado correctamente ✓")


if __name__ == "__main__":
    print("=== Tests GPU ===")
    test_remote_module_loads()
    test_gpu_method_exists()
    test_gpu_status()
    test_resolve_mode()
    test_detect_local_cuda()
    test_ssh_connection()
    print("\n✅ Tests GPU completados")
