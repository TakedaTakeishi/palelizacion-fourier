import pygame
import numpy as np
from display.states.state import AppState
from display.components.fourier_plot import FourierPlot
from core.types import (
    FUNC_NAMES, FUNC_SHORT_NAMES, METHOD_NAMES, METHOD_COLORS,
    FUNC_X4, FUNC_SQUARE, FUNC_SAWTOOTH, FUNC_TRIANGLE,
    function_real, fourier_approximation, fourier_terms_individual,
)
from core.orchestrator import Orchestrator

ALL_FUNCS = [FUNC_X4, FUNC_SQUARE, FUNC_SAWTOOTH, FUNC_TRIANGLE]

class SelectState(AppState):
    def __init__(self, orchestrator: Orchestrator, func_type=0, num_terms=30):
        super().__init__()
        self.orchestrator = orchestrator
        self.func_type = func_type
        self.num_terms = num_terms
        self.plot = FourierPlot(font_size=14)
        self.preview = self._make_preview()
        self.last_time = pygame.time.get_ticks() / 1000.0
        self._key_repeat_last = 0.0
        self._key_repeat_first = 0.35
        self._key_repeat_int = 0.05

        self._title_font = pygame.font.SysFont("arial", 30, bold=True)
        self._ctrl_font = pygame.font.SysFont("arial", 18)
        self._big_font = pygame.font.SysFont("arial", 22, bold=True)
        self._med_font = pygame.font.SysFont("consolas", 18)
        self._small_font = pygame.font.SysFont("consolas", 14)

    def _make_preview(self):
        max_t = min(self.num_terms, 200)
        x = np.linspace(-np.pi, np.pi, 400)
        f_real = function_real(x, self.func_type)
        terms = fourier_terms_individual(x, 200, self.func_type)
        approx = fourier_approximation(x, self.num_terms, self.func_type)
        return {"x": x, "f_real": f_real, "terms": terms[:max_t], "approx": approx}

    def enter(self):
        super().enter()
        self.preview = self._make_preview()
        self.last_time = pygame.time.get_ticks() / 1000.0
        self._key_repeat_last = 0.0

    def _step_terms(self, delta):
        new_val = self.num_terms + delta
        new_val = max(1, min(10000, new_val))
        if new_val != self.num_terms:
            self.num_terms = new_val
            self.preview = self._make_preview()

    def _set_function(self, idx):
        ft = ALL_FUNCS[idx]
        if ft != self.func_type:
            self.func_type = ft
            self.preview = self._make_preview()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.done = True
                self.next_state = "standby"
            elif event.key == pygame.K_RETURN:
                self.done = True
                self.next_state = "demo"
            elif event.key == pygame.K_1: self._set_function(0)
            elif event.key == pygame.K_2: self._set_function(1)
            elif event.key == pygame.K_3: self._set_function(2)
            elif event.key == pygame.K_4: self._set_function(3)
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_UP, pygame.K_RIGHT):
                self._step_terms(5); self._key_repeat_last = 0.0
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_DOWN, pygame.K_LEFT):
                self._step_terms(-5); self._key_repeat_last = 0.0

    def update(self):
        now = pygame.time.get_ticks() / 1000.0
        dt = now - self.last_time
        self.last_time = now
        self._handle_key_hold(dt)

    def _handle_key_hold(self, dt):
        for key, held in list(self.key_held.items()):
            if not held:
                continue
            self.key_held_timer[key] = self.key_held_timer.get(key, 0.0) + dt
            if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_UP, pygame.K_RIGHT):
                if self.key_held_timer[key] > self._key_repeat_first:
                    used = int((self.key_held_timer[key] - self._key_repeat_last) / self._key_repeat_int)
                    if used >= 1:
                        self._step_terms(5 * used)
                        self._key_repeat_last = self.key_held_timer[key]
            elif key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_DOWN, pygame.K_LEFT):
                if self.key_held_timer[key] > self._key_repeat_first:
                    used = int((self.key_held_timer[key] - self._key_repeat_last) / self._key_repeat_int)
                    if used >= 1:
                        self._step_terms(-5 * used)
                        self._key_repeat_last = self.key_held_timer[key]

    def render(self, surface):
        w, h = surface.get_size()
        surface.fill((5, 5, 14))

        title = self._title_font.render("Configurar demostración", True, (200, 220, 255))
        surface.blit(title, (w//2 - title.get_width()//2, 15))

        plot_rect = pygame.Rect(int(w * 0.04), int(h * 0.14),
                                int(w * 0.55), int(h * 0.65))
        has_disc = self.func_type in (FUNC_SQUARE, FUNC_SAWTOOTH, FUNC_TRIANGLE)
        self.plot.render(surface, plot_rect,
                         self.preview["x"], self.preview["f_real"],
                         self.preview["approx"], glow=True,
                         title=f"Preview: {FUNC_NAMES[self.func_type]}",
                         has_discontinuity=has_disc)

        info_rect = pygame.Rect(int(w * 0.62), int(h * 0.14),
                                int(w * 0.34), int(h * 0.65))
        self._render_info(surface, info_rect, w, h)

        for txt in ["[ENTER] Iniciar  |  [1-4] Función  |  [+/-, Flechas] Términos  |  [ESC] Volver"]:
            s = self._ctrl_font.render(txt, True, (150, 200, 150))
            surface.blit(s, (w//2 - s.get_width()//2, h - 30))

    def _render_info(self, surface, rect, w, h):
        pygame.draw.rect(surface, (15, 15, 30), rect)
        pygame.draw.rect(surface, (60, 60, 90), rect, 1)

        x = rect.x + 20
        y = rect.y + 20
        s = self._big_font.render("Función a aproximar", True, (180, 220, 255))
        surface.blit(s, (x, y))
        y += 35

        for i, ft in enumerate(ALL_FUNCS):
            color = (255, 200, 100) if ft == self.func_type else (90, 90, 120)
            mark = ">>" if ft == self.func_type else "  "
            s = self._med_font.render(f"{mark} [{i+1}] {FUNC_SHORT_NAMES[ft]}", True, color)
            surface.blit(s, (x, y))
            y += 26

        y += 15
        s = self._big_font.render("Armónicos", True, (180, 220, 255))
        surface.blit(s, (x, y))
        y += 35

        nt_color = (255, 200, 100) if self.num_terms < 50 else (100, 255, 150)
        s = self._big_font.render(f"{self.num_terms}", True, nt_color)
        surface.blit(s, (x, y))
        y += 30

        y += 10
        s = self._small_font.render("Más armónicos = mejor aproximación", True, (120, 120, 150))
        surface.blit(s, (x, y))
        y += 20
        s = self._small_font.render("pero también = más cálculo", True, (120, 120, 150))
        surface.blit(s, (x, y))

        y += 30
        s = self._big_font.render("Métodos a comparar", True, (180, 220, 255))
        surface.blit(s, (x, y))
        y += 30
        for m in self.orchestrator.disponible():
            color = METHOD_COLORS.get(m.key(), (200, 200, 200))
            name = m.name()
            if m.key() == "gpu":
                st = self.orchestrator.gpu_status()
                if st["mode"] == "local":
                    name += " ● (Local)"
                elif st["connected"]:
                    name += " ● (Servidor)"
                else:
                    name += " ○ (No conectado)"
                    color = (120, 60, 60)
            s = self._med_font.render(f"  - {name}", True, color)
            surface.blit(s, (x, y))
            y += 22
