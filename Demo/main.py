import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from display.app import App

def main():
    import pygame
    pygame.display.init()
    info = pygame.display.Info()
    w, h = info.current_w, info.current_h
    app = App(width=w, height=h, fullscreen=True)
    app.run()

if __name__ == "__main__":
    main()
