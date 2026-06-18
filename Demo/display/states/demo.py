import pygame
import numpy as np
import math
from display.states.state import AppState

def get_displayed_term(anim_term, num_terms, saved_terms):
    if saved_terms <= 1 or num_terms <= 1:
        return num_terms
    if anim_term >= saved_terms:
        return num_terms
    ratio = (anim_term - 1) / (saved_terms - 1)
    val = int(np.exp(ratio * np.log(num_terms)))
    return max(1, min(num_terms, val))
from display.components.fourier_plot import FourierPlot
from display.components.bar_chart import BarChart
from display.components.code_panel import CodePanel
from core.types import FUNC_NAMES, METHOD_NAMES, METHOD_COLORS, a0_func

PHASE_RUNNING = 0
PHASE_ANIMATING = 1
PHASE_COMPARISON = 2
PHASE_DONE = 3
PHASE_SHOW_RESULT = 4

class DemoState(AppState):
    def __init__(self, orchestrator, func_type=0, num_terms=50):
        super().__init__()
        self.orchestrator = orchestrator
        self.func_type = func_type
        self.num_terms = min(num_terms, 20_000_000_000)
        self.plot = FourierPlot(font_size=14)
        self.chart = BarChart(font_size=14)
        self.code_panel = CodePanel(font_size=13)
        self.results = []
        self.current_idx = 0
        self.phase = PHASE_RUNNING
        self.methods = orchestrator.disponible()
        self.anim_term = 0
        self.phase_timer = 0
        self.last_time = pygame.time.get_ticks() / 1000.0
        self.method_function_map = {
            "secuencial": "main",
            "hilos": "rutina_hilo",
            "procesos": "main",
            "mpi": "main",
        }

        self._title_font = pygame.font.SysFont("arial", 30, bold=True)
        self._ctrl_font = pygame.font.SysFont("arial", 16)
        self._info_font = pygame.font.SysFont("consolas", 18)
        self._mini_title_font = pygame.font.SysFont("arial", 18, bold=True)
        self._mini_lbl_font = pygame.font.SysFont("consolas", 13, bold=True)
        self._mini_tiny_font = pygame.font.SysFont("consolas", 12)

    def enter(self):
        super().enter()
        self.current_idx = 0
        self.phase = PHASE_RUNNING
        self.results = []
        self.anim_term = 1
        self.phase_timer = 0
        self.last_time = pygame.time.get_ticks() / 1000.0

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.done = True
                self.next_state = "standby"
            elif event.key == pygame.K_SPACE:
                self.done = True
                self.next_state = "standby"
            elif event.key == pygame.K_r:
                self.enter()
            elif event.key == pygame.K_RETURN and self.phase == PHASE_DONE:
                self.enter()

    def update(self):
        now = pygame.time.get_ticks() / 1000.0
        dt = now - self.last_time
        self.last_time = now
        self.phase_timer += dt

        if self.phase == PHASE_RUNNING:
            if self.current_idx < len(self.methods):
                method = self.methods[self.current_idx]
                try:
                    result = method.run(self.func_type, self.num_terms)
                    self.results.append(result)
                except Exception as e:
                    print(f"Error en {method.name()}: {e}")
                    from core.types import ComputationResult
                    dummy = self.orchestrator.precomputar_standby(self.func_type, self.num_terms, 64)
                    result = ComputationResult(
                        method=method.key(), func_type=self.func_type,
                        num_terms=self.num_terms, elapsed=999.0,
                        x=dummy["x"], f_real=dummy["f_real"],
                        fourier_approx=dummy["approx"],
                        terms=dummy["terms"], csv_path=""
                    )
                    self.results.append(result)
                if method.key() == "mpi":
                    self.phase = PHASE_SHOW_RESULT
                    self.anim_term = self.num_terms
                else:
                    self.phase = PHASE_ANIMATING
                    self.anim_term = 1
                self.phase_timer = 0
                self.current_idx += 1
                self.last_time = pygame.time.get_ticks() / 1000.0
            else:
                self.phase = PHASE_COMPARISON
                self.phase_timer = 0

        elif self.phase == PHASE_SHOW_RESULT:
            if self.phase_timer > 1.0:
                self.phase = PHASE_ANIMATING
                self.anim_term = 1
                self.phase_timer = 0

        elif self.phase == PHASE_ANIMATING:
            saved_terms = min(self.num_terms, 100)
            anim_rate = saved_terms / 2.0  # Animate over 2.0 seconds
            self.anim_term += dt * anim_rate
            if self.anim_term >= saved_terms or self.phase_timer > 3.0:
                if self.current_idx < len(self.methods):
                    self.phase = PHASE_RUNNING
                else:
                    self.phase = PHASE_COMPARISON
                    self.phase_timer = 0
                self.phase_timer = 0

        elif self.phase == PHASE_COMPARISON:
            if self.phase_timer > 4.0:
                self.phase = PHASE_DONE
                self.phase_timer = 0

    def render(self, surface):
        w, h = surface.get_size()
        surface.fill((5, 5, 14))
        if self.phase == PHASE_COMPARISON or self.phase == PHASE_DONE:
            self._render_comparison(surface, w, h)
        else:
            self._render_step(surface, w, h)

    def _render_step(self, surface, w, h):
        plot_rect = pygame.Rect(int(w * 0.03), int(h * 0.12),
                                int(w * 0.55), int(h * 0.65))
        code_rect = pygame.Rect(int(w * 0.60), int(h * 0.12),
                                int(w * 0.37), int(h * 0.65))
        time_rect = pygame.Rect(int(w * 0.03), int(h * 0.79),
                                int(w * 0.94), int(h * 0.10))
        ctrl_rect = pygame.Rect(int(w * 0.03), int(h * 0.91),
                                int(w * 0.94), int(h * 0.06))

        if self.results:
            r = self.results[-1]
            method = self.methods[self.current_idx - 1] if self.current_idx > 0 else None
            name = METHOD_NAMES.get(r.method, r.method)
            color = METHOD_COLORS.get(r.method, (200, 200, 200))

            title = self._title_font.render(f">> {name}", True, color)
            surface.blit(title, (20, 10))

            saved_terms = min(r.num_terms, 100)
            anim_t = min(int(self.anim_term), saved_terms)
            a0 = a0_func(r.func_type)

            if self.anim_term >= saved_terms:
                approx = r.fourier_approx
                displayed_t = r.num_terms
            else:
                approx = a0 / 2 + np.sum(r.terms[:anim_t], axis=0)
                displayed_t = get_displayed_term(self.anim_term, r.num_terms, saved_terms)

            full_f_min = float(r.f_real.min())
            full_f_max = float(r.f_real.max())
            full_approx_all = a0 / 2 + np.sum(r.terms, axis=0)
            full_f_min = min(full_f_min, float(full_approx_all.min()))
            full_f_max = max(full_f_max, float(full_approx_all.max()))
            full_margin = (full_f_max - full_f_min) * 0.08 if full_f_max > full_f_min else 1.0
            step_y_range = (full_f_min - full_margin, full_f_max + full_margin)

            has_disc = r.func_type in (1, 2, 3)
            self.plot.render(surface, plot_rect, r.x, r.f_real, approx,
                             r.terms, anim_t, glow=True,
                             title=f"Aproximación armónica ({displayed_t:,}/{r.num_terms:,})",
                             has_discontinuity=has_disc, y_range=step_y_range)

            method_obj = self.methods[self.current_idx - 1] if self.current_idx > 0 else None
            src = method_obj.source_file() if method_obj else ""
            func_name = method_obj.function_to_display() if method_obj else None
            self.code_panel.render(surface, src, code_rect,
                                   func_name=func_name,
                                   title=f"src: {name}",
                                   max_lines=40)

            t_surf = self._info_font.render(f"Tiempo: {r.elapsed:.4f}s", True, (100, 255, 150))
            surface.blit(t_surf, (time_rect.x + 10, time_rect.y + 10))
            if self.results and self.results[0].elapsed > 0 and r.elapsed > 0:
                sp = self.results[0].elapsed / r.elapsed
                sp_surf = self._info_font.render(f"Speedup: {sp:.2f}x", True, (255, 200, 100))
                surface.blit(sp_surf, (time_rect.x + 280, time_rect.y + 10))
            method_idx = self.current_idx
            total = len(self.methods)
            prog_surf = self._info_font.render(f"Método {method_idx}/{total}", True, (150, 200, 255))
            surface.blit(prog_surf, (time_rect.right - prog_surf.get_width() - 10,
                                      time_rect.y + 10))

            for txt in ["[ESC] Volver  |  [R] Repetir  |  [SPACE] Pausar/Salir"]:
                s = self._ctrl_font.render(txt, True, (120, 120, 150))
                surface.blit(s, (ctrl_rect.centerx - s.get_width()//2, ctrl_rect.y + 5))

    def _render_comparison(self, surface, w, h):
        if not self.results:
            return

        title = self._title_font.render(
            "Comparación: Mismo resultado, diferente velocidad",
            True, (180, 220, 255))
        surface.blit(title, (w//2 - title.get_width()//2, 15))

        main_plot_rect = pygame.Rect(int(w * 0.03), int(h * 0.10),
                                    int(w * 0.40), int(h * 0.42))
        bar_rect = pygame.Rect(int(w * 0.45), int(h * 0.10),
                               int(w * 0.52), int(h * 0.42))
        mini_rect = pygame.Rect(int(w * 0.03), int(h * 0.55),
                                int(w * 0.94), int(h * 0.38))

        r0 = self.results[0]
        has_disc = r0.func_type in (1, 2, 3)
        full_f_min = min(float(r0.f_real.min()), float(r0.fourier_approx.min()))
        full_f_max = max(float(r0.f_real.max()), float(r0.fourier_approx.max()))
        full_margin = (full_f_max - full_f_min) * 0.08 if full_f_max > full_f_min else 1.0
        demo_y_range = (full_f_min - full_margin, full_f_max + full_margin)
        n_all = min(r0.num_terms, len(r0.terms))
        self.plot.render(surface, main_plot_rect, r0.x, r0.f_real, r0.fourier_approx,
                         terms=r0.terms, current_term=n_all,
                         glow=True,
                         title="Superposición de los 5 métodos (resultado idéntico)",
                         has_discontinuity=has_disc, y_range=demo_y_range)

        sp = r0.elapsed / min(r.elapsed for r in self.results) if self.results[0].elapsed > 0 else 1
        self.chart.render(surface, bar_rect, self.results, speedup=sp)

        self._render_mini_graphs(surface, mini_rect)

        msg = "[ESC] Volver a Standby  |  [R] Repetir  |  [ENTER] Reiniciar"
        s = self._ctrl_font.render(msg, True, (120, 200, 150))
        surface.blit(s, (w//2 - s.get_width()//2, h - 25))

    def _render_mini_graphs(self, surface, rect):
        title = self._mini_title_font.render(
            "5 métodos, mismo resultado:", True, (180, 220, 255))
        surface.blit(title, (rect.x, rect.y))

        gpu_label = "GPU (CUDA)"
        gpu_available = any(r.method == "gpu" for r in self.results)
        if not gpu_available:
            gpu_label = "GPU (no disponible)"

        all_methods = [
            ("secuencial", "Secuencial", (102, 252, 241)),
            ("hilos", "Hilos", (255, 0, 127)),
            ("procesos", "Procesos", (57, 255, 20)),
            ("mpi", "MPI", (255, 209, 102)),
            ("gpu", gpu_label, (113, 29, 154)),
        ]
        r0 = self.results[0]
        time_map = {r.method: r.elapsed for r in self.results}
        n_methods = len(all_methods)
        rows = 2
        cols = (n_methods + rows - 1) // rows
        cell_w = (rect.width - 10) // cols
        cell_h = (rect.height - 45) // rows
        start_x = rect.x + 5
        start_y = rect.y + 30
        for i, (key, label, color) in enumerate(all_methods):
            row = i // cols
            col = i % cols
            cx = start_x + col * cell_w
            cy = start_y + row * cell_h
            cell = pygame.Rect(cx, cy, cell_w - 6, cell_h - 6)
            pygame.draw.rect(surface, (18, 18, 32), cell)
            pygame.draw.rect(surface, color, cell, 1)

            if key in time_map:
                self._draw_mini_curve(surface, cell, r0.x, r0.f_real, r0.fourier_approx, color)
            else:
                pen = self._mini_tiny_font.render(
                    "post-MVP", True, (140, 140, 170))
                surface.blit(pen, (cell.centerx - pen.get_width()//2, cell.centery - 6))

            lbl = self._mini_lbl_font.render(
                label, True, color)
            surface.blit(lbl, (cell.x + 5, cell.y + 4))

            if key in time_map:
                t_txt = f"{time_map[key]*1000:.1f}ms"
                t_surf = self._mini_tiny_font.render(
                    t_txt, True, (220, 220, 240))
                surface.blit(t_surf, (cell.right - t_surf.get_width() - 5,
                                     cell.bottom - t_surf.get_height() - 4))

    def _draw_mini_curve(self, surface, cell, x, f_real, approx, color):
        inner = cell.inflate(-6, -22)
        if inner.width <= 0 or inner.height <= 0:
            return
        x_min, x_max = float(x.min()), float(x.max())
        if x_max == x_min:
            return
        f_min = float(np.min(f_real))
        f_max = float(np.max(f_real))
        approx_min = float(np.min(approx))
        approx_max = float(np.max(approx))
        y_min = min(f_min, approx_min)
        y_max = max(f_max, approx_max)
        if y_max - y_min < 0.1:
            y_max += 0.5
            y_min -= 0.5
        margin = (y_max - y_min) * 0.10
        y_min -= margin
        y_max += margin

        def to_screen(xx, yy):
            px = inner.left + (xx - x_min) / (x_max - x_min) * inner.width
            py = inner.bottom - (yy - y_min) / (y_max - y_min) * inner.height
            return int(px), int(py)

        real_pts = [to_screen(x[i], f_real[i]) for i in range(len(x))]
        try:
            pygame.draw.lines(surface, (200, 200, 210), False, real_pts, 1)
        except (TypeError, pygame.error):
            pass
        approx_pts = [to_screen(x[i], approx[i]) for i in range(len(x))]
        try:
            pygame.draw.lines(surface, color, False, approx_pts, 2)
        except (TypeError, pygame.error):
            pass
