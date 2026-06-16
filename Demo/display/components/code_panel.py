import os
import re
import pygame
import pygments
import pygments.lexers
import pygments.token

class CodePanel:
    def __init__(self, font_size=14):
        self.font_size = font_size
        self.font = pygame.font.SysFont("consolas", font_size)
        self.lexer = pygments.lexers.CLexer()
        self.bg_color = (10, 10, 18)
        self.border_color = (60, 60, 90)
        self.pad = 12
        self.token_styles = {
            pygments.token.Keyword:       (86, 156, 214),
            pygments.token.Keyword.Type:  (86, 156, 214),
            pygments.token.Name.Function: (220, 220, 170),
            pygments.token.String:        (206, 145, 120),
            pygments.token.Comment:       (87, 166, 74),
            pygments.token.Comment.Single:(87, 166, 74),
            pygments.token.Comment.Multiline:(87, 166, 74),
            pygments.token.Number:        (181, 206, 168),
            pygments.token.Operator:      (212, 212, 212),
            pygments.token.Punctuation:   (212, 212, 212),
            pygments.token.Name:          (212, 212, 212),
        }
        self.default_color = (212, 212, 212)
        self.line_height = font_size + 2

    def extract_region(self, source_path, func_name=None, max_lines=40):
        if not os.path.exists(source_path):
            return [f"// {source_path} no encontrado"]
        with open(source_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if not func_name:
            return text.split("\n")[:max_lines]
        pattern = re.compile(
            rf"^[\w\s\*]+{re.escape(func_name)}\s*\([^)]*\)\s*\{{",
            re.MULTILINE
        )
        m = pattern.search(text)
        if not m:
            return [f"// No se encontro {func_name}()"]
        start = m.start()
        depth = 0
        i = m.end() - 1
        end = i
        for i in range(m.end() - 1, min(len(text), m.end() + 8000)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        return text[start:end].split("\n")

    def render(self, surface, source_path, rect, func_name=None,
               scroll_offset=0, title=None, max_lines=40):
        pygame.draw.rect(surface, self.bg_color, rect)
        pygame.draw.rect(surface, self.border_color, rect, 1)

        if title:
            t = self.font.render(title, True, (180, 220, 255))
            surface.blit(t, (rect.x + self.pad, rect.y + self.pad))

        lines = self.extract_region(source_path, func_name, max_lines)
        y = rect.y + self.pad + self.line_height + (self.line_height // 2 if title else 0)
        x_start = rect.x + self.pad
        first = max(0, scroll_offset // self.line_height)
        line_num_width = 40

        for idx, line in enumerate(lines[first:first + 30]):
            ln = first + idx + 1
            num_surf = self.font.render(f"{ln:3d}", True, (80, 80, 100))
            surface.blit(num_surf, (x_start, y))

            try:
                tokens = list(self.lexer.get_tokens(line + "\n"))
            except Exception:
                tokens = [(pygments.token.Text, line)]
            x = x_start + line_num_width
            for ttype, value in tokens:
                if value == "\n":
                    continue
                color = self.default_color
                for cls, c in self.token_styles.items():
                    if ttype in cls:
                        color = c
                        break
                try:
                    img = self.font.render(value, True, color)
                    surface.blit(img, (x, y))
                    x += img.get_width()
                except pygame.error:
                    pass
            y += self.line_height
