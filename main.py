import sys
import os
import argparse

demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo")
sys.path.insert(0, demo_dir)

from display.app import App


def main():
    parser = argparse.ArgumentParser(description="Fourier Parallel Demo")
    parser.add_argument(
        "--gpu-mode", choices=["auto", "local", "remote"], default=None,
        help="Modo de ejecución GPU: auto (detecta), local, remote (SSH)"
    )
    args, _ = parser.parse_known_args()

    if args.gpu_mode:
        os.environ["GPU_MODE"] = args.gpu_mode

    import pygame

    pygame.display.init()
    info = pygame.display.Info()
    w, h = info.current_w, info.current_h
    app = App(width=w, height=h, fullscreen=True)
    app.run()


if __name__ == "__main__":
    main()
