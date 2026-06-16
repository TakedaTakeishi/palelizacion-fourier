import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Demo"))

from core.orchestrator import Orchestrator
from core.types import FUNC_X4, FUNC_SQUARE

def test_all_methods():
    o = Orchestrator()
    methods = o.disponible()
    assert len(methods) >= 2, f"Se esperaban al menos 2 métodos, hay {len(methods)}"

    for func_type, func_name in [(FUNC_X4, "x4"), (FUNC_SQUARE, "square")]:
        print(f"\nProbando función: {func_name}")
        results = o.correr_todos(func_type, 10)

        assert len(results) == len(methods), f"Deben haber {len(methods)} resultados"

        approx_first = results[0].fourier_approx
        for r in results[1:]:
            diff = abs(r.fourier_approx - approx_first).max()
            assert diff < 1e-10, (
                f"Diferencia entre {results[0].method} y {r.method}: {diff}"
            )

        print(f"  {len(results)} métodos producen el mismo resultado ✓")
        for r in results:
            print(f"    {r.method:20s}  {r.elapsed:.6f}s")

    print("\n✅ Todos los tests de integracion pasaron!")

if __name__ == "__main__":
    test_all_methods()
