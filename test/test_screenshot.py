"""
Captura screenshots de los distintos estados de la UI para revisión visual.
Las imágenes se sobrescriben en cada ejecución (no se acumulan).
Uso: uv run python test/test_screenshot.py
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Demo"))

import pygame
from display.app import App
from display.states.standby import StandbyState
from display.states.demo import DemoState

SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
os.makedirs(SHOT_DIR, exist_ok=True)

def shot(app, name):
    path = os.path.join(SHOT_DIR, f"{name}.png")
    pygame.image.save(app.screen, path)
    print(f"  Capturado: {path}")
    return path

def run_frames(state, app, n=5, dt=0.1):
    for _ in range(n):
        state.update()
        state.render(app.screen)
        pygame.display.flip()
        time.sleep(dt)

def post_key(state, key):
    state.handle_event(pygame.event.Event(pygame.KEYDOWN,
        {"key": key, "mod": 0, "unicode": "", "scancode": 0}))

def main():
    pygame.display.init()
    info = pygame.display.Info()
    w, h = min(1920, info.current_w), min(1080, info.current_h)
    app = App(width=w, height=h, fullscreen=False)
    print(f"Resolucion de captura: {w}x{h}")

    standby = app.current_state
    standby._set_function(0)
    run_frames(standby, app, n=10, dt=0.08)
    shot(app, "01_standby_x4")

    standby._set_function(1)
    run_frames(standby, app, n=10, dt=0.08)
    shot(app, "02_standby_square")

    standby._set_function(2)
    run_frames(standby, app, n=10, dt=0.08)
    shot(app, "03_standby_sawtooth")

    standby._set_function(3)
    run_frames(standby, app, n=10, dt=0.08)
    shot(app, "04_standby_triangle")

    print("\n[5/8] Capturando SELECT (square, 30 armónicos)...")
    from display.states.select import SelectState
    select = SelectState(app.orchestrator, func_type=1, num_terms=30)
    app.current_state = select
    run_frames(select, app, n=3, dt=0.05)
    shot(app, "05_select")

    print("\n[6/8] Capturando SELECT con más armónicos (50)...")
    select._step_terms(20)
    run_frames(select, app, n=3, dt=0.05)
    shot(app, "06_select_more_terms")

    print("\n[7/8] Capturando DEMO estado PHASE_RUNNING (x4)...")
    orch = app.orchestrator
    demo = DemoState(orch, func_type=0, num_terms=20)
    demo.enter()
    app.current_state = demo
    run_frames(demo, app, n=2, dt=0.05)
    shot(app, "07_demo_running")

    print("\n[8/8] Capturando DEMO estado COMPARISON (5 métodos)...")
    fake_results = []
    from core.types import ComputationResult
    import numpy as np
    base = orch.precomputar_standby(0, 20, 64)
    for method, t in [("secuencial", 0.025), ("hilos", 0.022), ("procesos", 0.21), ("mpi", 0.48)]:
        fake_results.append(ComputationResult(
            method=method, func_type=0, num_terms=20, elapsed=t,
            x=base["x"], f_real=base["f_real"], fourier_approx=base["approx"],
            terms=base["terms"], csv_path=""
        ))
    demo.results = fake_results
    demo.phase = 3
    demo.phase_timer = 0
    demo.render(app.screen)
    pygame.display.flip()
    time.sleep(0.2)
    shot(app, "08_demo_comparison")

    print(f"\nListo. Imagenes en: {SHOT_DIR}")
    pygame.quit()

if __name__ == "__main__":
    main()
