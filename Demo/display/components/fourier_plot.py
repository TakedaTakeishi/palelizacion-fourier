import numpy as np
import pygame

class FourierPlot:
    def __init__(self, font_size=14):
        self.bg_color = (10, 10, 18)
        self.grid_color = (35, 35, 60)
        self.axis_color = (80, 80, 110)
        self.label_color = (130, 130, 160)
        self.real_color = (240, 240, 240)
        self.approx_color = (255, 120, 80)
        self.term_color = (80, 80, 120)
        self.font = pygame.font.SysFont("consolas", font_size)
        self.pad_left = 70
        self.pad_right = 25
        self.pad_top = 35
        self.pad_bottom = 45

    def _auto_yrange(self, *arrays, f_real=None):
        all_vals = np.concatenate([a.ravel() for a in arrays if a is not None and len(a) > 0])
        if len(all_vals) == 0:
            return -12, 12
        y_min, y_max = float(all_vals.min()), float(all_vals.max())

        if f_real is not None and len(f_real) > 0:
            fr_min, fr_max = float(f_real.min()), float(f_real.max())
            fr_range = fr_max - fr_min
            if fr_range < 3.0:
                cap = max(1.6, fr_range * 0.6)
                y_min = max(y_min, fr_min - cap)
                y_max = min(y_max, fr_max + cap)

        if y_max - y_min < 0.5:
            y_max += 1
            y_min -= 1
        margin = (y_max - y_min) * 0.08
        return y_min - margin, y_max + margin


    def _nice_ticks(self, v_min, v_max, n=5):
        if v_max <= v_min:
            return []
        span = v_max - v_min
        raw = span / max(1, n)
        if raw <= 0:
            return []
        mag = 10 ** np.floor(np.log10(raw))
        norm = raw / mag
        if norm < 1.5: step = 1 * mag
        elif norm < 3.5: step = 2 * mag
        elif norm < 7.5: step = 5 * mag
        else: step = 10 * mag
        ticks = []
        v = np.ceil(v_min / step) * step
        while v <= v_max + 1e-9:
            ticks.append(v)
            v += step
        return ticks

    def render(self, surface, rect, x, f_real, approx, terms=None,
               current_term=None, glow=True, title=None, has_discontinuity=False,
               y_range=None):
        inner = pygame.Rect(
            rect.x + self.pad_left,
            rect.y + self.pad_top,
            max(1, rect.width - self.pad_left - self.pad_right),
            max(1, rect.height - self.pad_top - self.pad_bottom),
        )
        pygame.draw.rect(surface, self.bg_color, inner)

        if y_range is not None:
            y_min, y_max = y_range
        else:
            all_y = [f_real, approx]
            y_min, y_max = self._auto_yrange(*all_y, f_real=f_real)
        x_min, x_max = float(x.min()), float(x.max())
        if x_max == x_min:
            x_max = x_min + 1

        def to_screen(xx, yy):
            px = inner.left + (xx - x_min) / (x_max - x_min) * inner.width
            py = inner.bottom - (yy - y_min) / (y_max - y_min) * inner.height
            return int(px), int(py)

        for ty in self._nice_ticks(y_min, y_max):
            py = to_screen(x_min, ty)[1]
            if inner.top <= py <= inner.bottom:
                pygame.draw.line(surface, self.grid_color,
                                 (inner.left, py), (inner.right, py), 1)
                lbl = self.font.render(f"{ty:+.0f}", True, self.label_color)
                surface.blit(lbl, (rect.x + 5, py - 7))

        for tx in self._nice_ticks(x_min, x_max, 5):
            px = to_screen(tx, y_min)[0]
            if inner.left <= px <= inner.right:
                pygame.draw.line(surface, self.grid_color,
                                 (px, inner.top), (px, inner.bottom), 1)
                lbl = self.font.render(f"{tx:+.1f}", True, self.label_color)
                surface.blit(lbl, (px - 12, inner.bottom + 6))

        pygame.draw.line(surface, (50, 50, 75),
                         (inner.left, to_screen(0, y_min)[1]),
                         (inner.right, to_screen(0, y_min)[1]), 1)
        pygame.draw.line(surface, (50, 50, 75),
                         (to_screen(0, y_min)[0], inner.top),
                         (to_screen(0, y_min)[0], inner.bottom), 1)

        prev_clip = surface.get_clip()
        surface.set_clip(inner)

        if terms is not None and current_term is not None:
            n_termas = min(current_term, len(terms))
            if n_termas > 20:
                draw_count = min(n_termas, 30)
                term_surf = pygame.Surface((inner.width, inner.height), pygame.SRCALPHA)
                for i in range(draw_count):
                    pts = [to_screen(x[j], terms[i][j]) for j in range(len(x))]
                    try:
                        pygame.draw.lines(term_surf, (90, 90, 130, 50), False, pts, 1)
                    except (TypeError, pygame.error):
                        pass
                surface.blit(term_surf, (inner.left, inner.top))
            else:
                for i in range(n_termas):
                    self._draw_curve_alpha(surface, inner, x, terms[i],
                                           (90, 90, 130), 60, 1, to_screen,
                                           break_on_discontinuity=has_discontinuity)

        if glow:
            self._draw_glow(surface, inner, x, f_real, self.real_color, to_screen,
                            break_on_discontinuity=has_discontinuity)
            self._draw_glow(surface, inner, x, approx, self.approx_color, to_screen,
                            break_on_discontinuity=has_discontinuity)

        self._draw_curve_alpha(surface, inner, x, f_real, self.real_color, 255, 2, to_screen,
                               break_on_discontinuity=has_discontinuity)
        self._draw_curve_alpha(surface, inner, x, approx, self.approx_color, 255, 3, to_screen,
                               break_on_discontinuity=has_discontinuity)

        surface.set_clip(prev_clip)
        pygame.draw.rect(surface, self.axis_color, inner, 1)

        if title:
            t = self.font.render(title, True, (180, 220, 255))
            surface.blit(t, (inner.left + 5, inner.top + 5))

    def _draw_glow(self, surface, rect, x, y_vals, color, to_screen, break_on_discontinuity=False):
        for width, alpha in [(5, 30)]:
            self._draw_curve_alpha(surface, rect, x, y_vals, color, alpha, width, to_screen,
                                   break_on_discontinuity=break_on_discontinuity)

    def _draw_curve_alpha(self, surface, rect, x, y_vals, color, alpha, width, to_screen,
                          break_threshold=0.9, break_on_discontinuity=False):
        if len(x) != len(y_vals) or len(x) < 2:
            return
        if isinstance(color, tuple) and len(color) == 3:
            col = (*color, alpha)
        else:
            col = color

        if break_on_discontinuity:
            breaks = [0]
            for i in range(1, len(x)):
                if y_vals[i] is not None and y_vals[i - 1] is not None:
                    if abs(y_vals[i] - y_vals[i - 1]) > break_threshold:
                        breaks.append(i)
            breaks.append(len(x))
        else:
            breaks = [0, len(x)]

        ranges = []
        for i in range(len(breaks) - 1):
            r_start, r_end = breaks[i], breaks[i + 1]
            if r_end - r_start >= 2:
                ranges.append((r_start, r_end))

        for r_start, r_end in ranges:
            points = [to_screen(x[i], y_vals[i]) for i in range(r_start, r_end)]
            try:
                if alpha < 255:
                    size = (rect.width + width * 2, rect.height + width * 2)
                    surf = pygame.Surface(size, pygame.SRCALPHA)
                    offset_x = rect.left - width
                    offset_y = rect.top - width
                    shifted = [(p[0] - offset_x, p[1] - offset_y) for p in points]
                    pygame.draw.lines(surf, col, False, shifted, width)
                    surface.blit(surf, (offset_x, offset_y))
                else:
                    pygame.draw.lines(surface, col, False, points, width)
            except (TypeError, pygame.error):
                pass
