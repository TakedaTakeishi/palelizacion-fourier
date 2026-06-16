"""
Test that actually runs the demo through all 4 methods, including real MPI.
Captures the PHASE_RUNNING screen for each method.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Demo"))

import pygame
from display.app import App
from display.states.demo import DemoState

SHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shots")
os.makedirs(SHOT_DIR, exist_ok=True)

def shot(app, name):
    path = os.path.join(SHOT_DIR, f"{name}.png")
    pygame.image.save(app.screen, path)
    print(f"  Capturado: {path}")
    return path

def main():
    pygame.display.init()
    info = pygame.display.Info()
    w, h = min(1920, info.current_w), min(1080, info.current_h)
    app = App(width=w, height=h, fullscreen=False)
    print(f"Resolucion de captura: {w}x{h}")

    orch = app.orchestrator
    methods = orch.disponible()
    print(f"Metodos disponibles: {[m.name() for m in methods]}")

    demo = DemoState(orch, func_type=0, num_terms=20)
    demo.enter()
    app.current_state = demo

    last_method = None
    for cycle in range(60):
        demo.update()
        demo.render(app.screen)
        pygame.display.flip()
        time.sleep(0.05)
        current_method = None
        if demo.current_idx > 0 and demo.current_idx <= len(methods):
            current_method = methods[demo.current_idx - 1].name()
        elif demo.results:
            current_method = "comparison"
        if current_method != last_method:
            if current_method is not None:
                safe_name = current_method.replace(" ", "_").replace("(", "").replace(")", "")
                shot(app, f"seq_{current_method}_{demo.phase}")
            last_method = current_method
        if demo.phase == 3 and demo.phase_timer > 2.0:
            break

    print(f"\nListo. Imagenes en: {SHOT_DIR}")
    pygame.quit()

if __name__ == "__main__":
    main()
