import numpy as np
import pygame
from core.types import METHOD_COLORS, METHOD_NAMES

class BarChart:
    def __init__(self, font_size=14):
        self.bg_color = (10, 10, 18)
        self.axis_color = (80, 80, 110)
        self.label_color = (150, 150, 180)
        self.grid_color = (35, 35, 60)
        self.font = pygame.font.SysFont("consolas", font_size)
        self.font_big = pygame.font.SysFont("consolas", font_size + 4, bold=True)
        self.pad_left = 70
        self.pad_right = 20
        self.pad_top = 50
        self.pad_bottom = 50

    def _nice_ticks(self, v_max, n=5):
        if v_max <= 0:
            return []
        mag = 10 ** int(np.floor(np.log10(v_max)))
        norm = v_max / mag
        if norm < 1.5:
            step = 0.3 * mag
        elif norm < 3.5:
            step = 0.5 * mag
        else:
            step = mag
        ticks = []
        v = 0
        while v <= v_max * 1.05:
            ticks.append(v)
            v += step
        return ticks

    def render(self, surface, rect, results, speedup=None, title="Tiempos de ejecucion"):
        inner = pygame.Rect(
            rect.x + self.pad_left,
            rect.y + self.pad_top,
            rect.width - self.pad_left - self.pad_right,
            rect.height - self.pad_top - self.pad_bottom,
        )
        pygame.draw.rect(surface, self.bg_color, inner)
        pygame.draw.rect(surface, self.axis_color, inner, 1)

        if not results:
            return

        if title:
            t = self.font_big.render(title, True, (180, 220, 255))
            surface.blit(t, (rect.x + 15, rect.y + 12))

        times = [r.elapsed for r in results]
        v_max = max(times) * 1.15
        n = len(results)
        bar_gap = 18
        bar_w = max(40, min(80, (inner.width - bar_gap * (n + 1)) // n))
        total_w = n * bar_w + (n - 1) * bar_gap
        start_x = inner.left + (inner.width - total_w) // 2

        for ty in self._nice_ticks(v_max):
            py = inner.bottom - int(ty / v_max * inner.height)
            if inner.top <= py <= inner.bottom:
                pygame.draw.line(surface, self.grid_color,
                                 (inner.left, py), (inner.right, py), 1)
                lbl = self.font.render(f"{ty:.2f}s", True, self.label_color)
                surface.blit(lbl, (rect.x + 5, py - 8))

        for i, (r, t) in enumerate(zip(results, times)):
            color = METHOD_COLORS.get(r.method, (200, 200, 200))
            bar_h = int((t / v_max) * inner.height) if v_max > 0 else 0
            bx = start_x + i * (bar_w + bar_gap)
            by = inner.bottom - bar_h

            for glow_w, glow_a in [(bar_w + 4, 25)]:
                gh = inner.bottom - by
                if gh <= 0:
                    continue
                glow_surf = pygame.Surface((glow_w, gh), pygame.SRCALPHA)
                glow_surf.fill((*color, glow_a))
                surface.blit(glow_surf, (bx - (glow_w - bar_w) // 2, by))

            pygame.draw.rect(surface, color, (bx, by, bar_w, bar_h))
            pygame.draw.rect(surface, (255, 255, 255), (bx, by, bar_w, bar_h), 1)

            name = METHOD_NAMES.get(r.method, r.method)
            short = name.split()[0] if name else "?"
            name_surf = self.font.render(short, True, (220, 220, 240))
            surface.blit(name_surf, (bx + bar_w // 2 - name_surf.get_width() // 2,
                                     inner.bottom + 8))

            t_surf = self.font_big.render(f"{t:.4f}s", True, color)
            surface.blit(t_surf, (bx + bar_w // 2 - t_surf.get_width() // 2,
                                   by - t_surf.get_height() - 4))

        if speedup:
            txt = f"Speedup: {speedup:.1f}x vs Secuencial"
            sp = self.font.render(txt, True, (100, 255, 150))
            surface.blit(sp, (inner.left, rect.y + 32))
