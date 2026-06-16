import pygame
import numpy as np
from display.states.state import AppState
from display.components.fourier_plot import FourierPlot
from core.types import (
    FUNC_NAMES, FUNC_SHORT_NAMES, METHOD_NAMES, METHOD_COLORS,
    FUNC_X4, FUNC_SQUARE, FUNC_SAWTOOTH, FUNC_TRIANGLE,
    function_real, fourier_approximation, fourier_terms_individual, a0_func,
)
from core.orchestrator import Orchestrator, METHOD_GPU

ALL_FUNCS = [FUNC_X4, FUNC_SQUARE, FUNC_SAWTOOTH, FUNC_TRIANGLE]
MAX_TERMINOS = 10000
SPEED_LEVELS = [0.5, 1.0, 1.5, 3.0, 6.0, 12.0]

class StandbyState(AppState):
    PRECOMPUTED = {}

    def __init__(self, orchestrator: Orchestrator):
        super().__init__()
        self.orchestrator = orchestrator
        self.plot = FourierPlot(font_size=14)
        self.clock = pygame.time.Clock()

        self.func_idx = 0
        self.num_terms = 50
        self.display_terms = 1
        self.current_func = ALL_FUNCS[0]
        self.timer = 0
        self.func_switch_timer = 0
        self.FUNC_SWITCH_INTERVAL = 14.0
        self.last_time = pygame.time.get_ticks() / 1000.0
        self._key_repeat_last = 0.0
        self._key_repeat_first = 0.35
        self._key_repeat_int = 0.06
        self.speed_idx = 2
        self._last_display_t = -1
        self._cached_approx = np.empty(0)

        self._title_font = pygame.font.SysFont("arial", 32, bold=True)
        self._info_font = pygame.font.SysFont("consolas", 16)
        self._speed_font = pygame.font.SysFont("consolas", 16, bold=True)
        self._footer_font = pygame.font.SysFont("arial", 18)

        self._precompute_all()
        self.data = self._precompute()

    @classmethod
    def _precompute_all(cls):
        if cls.PRECOMPUTED:
            return
        x = np.linspace(-np.pi, np.pi, 500)
        for ft in ALL_FUNCS:
            f_real = function_real(x, ft)
            terms = fourier_terms_individual(x, 200, ft)
            approx = fourier_approximation(x, 200, ft)
            cumsum = np.cumsum(terms, axis=0)
            y_min = min(float(f_real.min()), float(approx.min()))
            y_max = max(float(f_real.max()), float(approx.max()))
            margin = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
            cls.PRECOMPUTED[ft] = {
                "x": x, "f_real": f_real, "terms": terms, "approx": approx,
                "cumsum": cumsum, "a0_half": a0_func(ft) / 2.0,
                "y_range": (y_min - margin, y_max + margin),
            }

    def _precompute(self):
        StandbyState._precompute_all()
        return StandbyState.PRECOMPUTED[self.current_func]

    def enter(self):
        super().enter()
        self.display_terms = 1
        self.func_switch_timer = 0
        self._key_repeat_last = 0.0
        self.data = self._precompute()

    def _set_function(self, idx):
        if idx == self.func_idx:
            return
        self.func_idx = idx
        self.current_func = ALL_FUNCS[idx]
        self.data = self._precompute()
        self.display_terms = 1

    def _step_terms(self, delta):
        new_val = self.num_terms + delta
        new_val = max(1, min(MAX_TERMINOS, new_val))
        if new_val != self.num_terms:
            self.num_terms = new_val
            self.display_terms = 1

    def _cycle_speed(self, direction):
        self.speed_idx = (self.speed_idx + direction) % len(SPEED_LEVELS)
        self._key_repeat_last = 0.0

    def _speed(self):
        return SPEED_LEVELS[self.speed_idx]

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                self.done = True
                self.next_state = "select"
                return
            if event.key == pygame.K_1: self._set_function(0)
            elif event.key == pygame.K_2: self._set_function(1)
            elif event.key == pygame.K_3: self._set_function(2)
            elif event.key == pygame.K_4: self._set_function(3)
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self._step_terms(5); self._key_repeat_last = 0.0
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self._step_terms(-5); self._key_repeat_last = 0.0
            elif event.key in (pygame.K_UP, pygame.K_RIGHT):
                self._step_terms(5); self._key_repeat_last = 0.0
            elif event.key in (pygame.K_DOWN, pygame.K_LEFT):
                self._step_terms(-5); self._key_repeat_last = 0.0
            elif event.key in (pygame.K_PAGEUP, pygame.K_F5, pygame.K_F6, pygame.K_F7):
                self._cycle_speed(1)
            elif event.key in (pygame.K_PAGEDOWN, pygame.K_F8, pygame.K_F9, pygame.K_F10):
                self._cycle_speed(-1)
            elif event.key == pygame.K_s:
                self._cycle_speed(1)
            elif event.key == pygame.K_F2:
                self._set_function((self.func_idx + 1) % len(ALL_FUNCS))

    def update(self):
        now = pygame.time.get_ticks() / 1000.0
        dt = now - self.last_time
        self.last_time = now

        self._handle_key_hold(dt)

        self.display_terms += dt * 6 * self._speed()
        if self.display_terms > self.num_terms:
            self.display_terms = 1

        self.func_switch_timer += dt
        if self.func_switch_timer > self.FUNC_SWITCH_INTERVAL:
            self.func_switch_timer = 0
            self.func_idx = (self.func_idx + 1) % len(ALL_FUNCS)
            self.current_func = ALL_FUNCS[self.func_idx]
            self.data = self._precompute()
            self.display_terms = 1

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
            elif key in (pygame.K_PAGEUP, pygame.K_F5, pygame.K_F6, pygame.K_F7):
                if self.key_held_timer[key] > self._key_repeat_first:
                    self._cycle_speed(1)
            elif key in (pygame.K_PAGEDOWN, pygame.K_F8, pygame.K_F9, pygame.K_F10):
                if self.key_held_timer[key] > self._key_repeat_first:
                    self._cycle_speed(-1)

    def render(self, surface):
        w, h = surface.get_size()
        surface.fill((5, 5, 14))

        plot_rect = pygame.Rect(int(w * 0.03), int(h * 0.10),
                                int(w * 0.94), int(h * 0.78))
        t = int(self.display_terms)
        if t != self._last_display_t:
            self._last_display_t = t
            n = min(t, len(self.data["terms"]))
            if n > 0:
                self._cached_approx = self.data["a0_half"] + self.data["cumsum"][n - 1]
            else:
                self._cached_approx = np.full_like(self.data["x"], self.data["a0_half"])
        approx = self._cached_approx
        n_terms_to_show = min(t, len(self.data["terms"]))
        has_disc = self.current_func in (FUNC_SQUARE, FUNC_SAWTOOTH, FUNC_TRIANGLE)
        self.plot.render(
            surface, plot_rect,
            self.data["x"], self.data["f_real"],
            approx, self.data["terms"], n_terms_to_show, glow=True,
            has_discontinuity=has_disc,
            y_range=self.data["y_range"],
        )

        self._render_header(surface, w)
        self._render_footer(surface, w, h)

    def _render_header(self, surface, w):
        title = self._title_font.render("FOURIER PARALLEL DEMO", True, (200, 220, 255))
        surface.blit(title, (20, 12))

        x = w - 20
        for i in range(4):
            color = (255, 200, 100) if ALL_FUNCS[i] == self.current_func else (90, 90, 120)
            lbl = f"[{i+1}] {FUNC_SHORT_NAMES[ALL_FUNCS[i]]}"
            surf = self._info_font.render(lbl, True, color)
            x -= surf.get_width() + 20
            surface.blit(surf, (x, 24))

        ft = FUNC_SHORT_NAMES[self.current_func]
        nt = int(self.display_terms)
        lbl = f"f(x): {ft}    |    Armónicos: {nt}/{self.num_terms}"
        info = self._info_font.render(lbl, True, (180, 180, 220))
        surface.blit(info, (20, 52))

        speed_label = self._info_font.render("Velocidad:", True, (180, 180, 220))
        surface.blit(speed_label, (20, 76))
        sx = 20 + speed_label.get_width() + 10
        for i, level in enumerate(SPEED_LEVELS):
            if i == self.speed_idx:
                txt = self._speed_font.render(f" [{level}x] ", True, (255, 220, 100))
                pygame.draw.rect(surface, (60, 60, 90),
                                 (sx - 2, 74, txt.get_width() + 4, txt.get_height() + 4), 1)
            else:
                txt = self._info_font.render(f" {level}x ", True, (100, 100, 140))
            surface.blit(txt, (sx, 76))
            sx += txt.get_width() + 4

        gpu_st = self.orchestrator.gpu_status()
        if gpu_st["mode"] == "local":
            dot = self._info_font.render("● GPU: Local", True, (100, 255, 100))
        elif gpu_st["connected"]:
            dot = self._info_font.render("● GPU: Servidor", True, (100, 255, 100))
        elif gpu_st["mode"] == "unavailable":
            dot = self._info_font.render("● GPU: No disponible", True, (200, 100, 100))
        else:
            dot = self._info_font.render("● GPU: Sin conexión", True, (200, 100, 100))
        surface.blit(dot, (sx + 20, 76))

    def _render_footer(self, surface, w, h):
        items = [
            ("ESPACIO", "Demo", (255, 100, 100)),
            ("1-4", "Función", (100, 200, 255)),
            ("+/-", "Armónicos", (100, 200, 255)),
            ("S/PgUp/Dn", "Velocidad", (100, 200, 255)),
            ("ESC", "Salir", (200, 100, 100)),
        ]
        x = 20
        y = h - 30
        for key, desc, color in items:
            k_surf = self._footer_font.render(f"[{key}]", True, color)
            d_surf = self._footer_font.render(f" {desc}", True, (150, 150, 180))
            surface.blit(k_surf, (x, y))
            surface.blit(d_surf, (x + k_surf.get_width(), y))
            x += k_surf.get_width() + d_surf.get_width() + 30
