# Fourier Parallel Demo

Demo interactiva para stand de Expo que muestra 4 formas de paralelismo en una sola computadora, usando aproximación de Series de Fourier como caso de estudio visual.

## Métodos de Paralelismo

| # | Método | Archivo | Paradigma |
|---|---|---|---|
| 1 | Secuencial | `1_Secuencial/fourier_seq.c` | Loop simple, un solo núcleo |
| 2 | Hilos | `4_Hilos/hilos_fourier.c` | pthreads, 4 hilos |
| 3 | Procesos | `3_Procesos/fourier.c` | fork() + memoria compartida System V |
| 4 | MPI | `5_MPI/fourier_mpi.c` | Paso de mensajes, 4 procesos |

*(GPU/CUDA planeado para post-MVP)*

## Funciones Soportadas

| ID | Función | Serie de Fourier |
|---|---|---|
| 0 | `f(x) = x⁴ − 3x` | Coeficientes analíticos pre-calculados |
| 1 | Square wave | `(4/π) Σ sin((2k-1)x)/(2k-1)` |
| 2 | Sawtooth wave | `2 Σ (-1)^(n+1) sin(nx)/n` |
| 3 | Triangle wave | `(8/π²) Σ (-1)^k sin((2k+1)x)/(2k+1)²` |

## Uso

### Modo Demo (público)
```bash
cd Demo
uv run python main.py
```

### Modo Comparativa (solo datos)
```bash
cd Comparativa
pixi run python src/comparativa_fourier.py
```

### Ejecutar programas C individualmente
```bash
# Windows
1_Secuencial/fourier_seq.exe --func 0 --terms 50
4_Hilos/hilos_fourier.exe --func 1 --terms 10

# WSL/Linux
cd 3_Procesos && ./fourier --func 0 --terms 50
cd 5_MPI && mpirun -np 4 ./fourier_mpi --func 2 --terms 20
```

## Tests
```bash
# Compilación
uv run --directory Demo python test/test_compilation.py

# Integración (resultados idénticos entre métodos)
uv run --directory Demo python test/test_integration.py
```

## Pendientes (post-MVP)

| Prioridad | Tarea | Detalle |
|---|---|---|
| Alta | **GPU/CUDA** | Modificar `7_CUDA/fourier.cu` para `--func --terms`, integrar en orquestador |
| Alta | **GPU remota** | Cliente/servidor TCP para GPU por SSH cuando no hay GPU local |
| Media | **Selector de función completo** | Habilitar las 4 funciones en los programas C (ya en `fourier_core.h`, probar sawtooth y triangle) |
| Media | **Tests de UI** | Capturar screenshots con `pygame.image.save()` y comparar contra baseline usando SSIM o modelo de visión |
| Media | **Afinar UI** | Layout responsive 2K, transiciones más pulidas, animaciones de partículas en standby |
| Baja | **Sockets** | Implementar método de paralelismo vía sockets (ya hay datos de referencia en `Comparativa/data/Con_sockets.csv`) |

## Arquitectura (Patrones de Diseño)

- **Strategy Pattern**: Cada método de paralelismo (`ComputationMethod`) implementa su propio `run()`
- **State Pattern**: La app Pygame alterna entre `StandbyState` y `DemoState`
- **Template Method**: La secuencia de demo tiene estructura fija (ejecutar → animar → comparar)
- **Shared Header**: Todos los programas C comparten `common/fourier_core.h`

## Dependencias

### C
- gcc, make, pthreads, math.h
- OpenMPI (para MPI)
- CUDA Toolkit (para GPU, post-MVP)

### Python
- Python 3.11+ (vía uv)
- pygame, numpy, pandas, pyyaml, pygments
