import pygame
import sys
import math
from core.orchestrator import Orchestrator
from display.states.standby import StandbyState
from display.states.demo import DemoState
from display.states.select import SelectState

class App:
    def __init__(self, width=1920, height=1080, fullscreen=True):
        pygame.init()
        pygame.display.set_caption("Fourier Parallel Demo")
        flags = pygame.FULLSCREEN | pygame.DOUBLEBUF | pygame.HWSURFACE if fullscreen else 0
        self.screen = pygame.display.set_mode((width, height), flags)
        self.clock = pygame.time.Clock()
        self.running = True
        self.orchestrator = Orchestrator()
        self.standby = StandbyState(self.orchestrator)
        self.current_state = self.standby
        self.current_state.enter()
        self._k_buf = []
        self._diag_mode = False
        self._diag_timer = 0.0

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if self._diag_mode:
                        self._diag_mode = False
                        self._k_buf.clear()
                    elif type(self.current_state).__name__ == "StandbyState":
                        self.running = False
                    else:
                        self.current_state.done = True
                        self.current_state.next_state = "standby"
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F4:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_ALT:
                        self.running = False
                elif event.type == pygame.KEYDOWN:
                    self._k_buf.append(event.key)
                    if len(self._k_buf) > 10:
                        self._k_buf.pop(0)
                    _target = [pygame.K_UP, pygame.K_UP, pygame.K_DOWN, pygame.K_DOWN,
                               pygame.K_LEFT, pygame.K_RIGHT, pygame.K_LEFT, pygame.K_RIGHT,
                               pygame.K_b, pygame.K_a]
                    if self._k_buf == _target:
                        self._diag_mode = not self._diag_mode
                        self._k_buf.clear()
                        continue
                    
                    if self._diag_mode:
                        continue

                    self.current_state.on_key_down(event.key)
                    self.current_state.handle_event(event)
                elif event.type == pygame.KEYUP:
                    if self._diag_mode:
                        continue
                    self.current_state.on_key_up(event.key)
                else:
                    self.current_state.handle_event(event)

            if self.current_state.done:
                next_name = self.current_state.next_state
                last_state = self.current_state
                self.current_state.exit()
                if next_name == "select":
                    ft = self.standby.current_func
                    nt = self.standby.num_terms
                    self.current_state = SelectState(self.orchestrator, ft, nt)
                elif next_name == "demo":
                    if isinstance(last_state, SelectState):
                        ft = last_state.func_type
                        nt = last_state.num_terms
                    else:
                        ft = self.standby.current_func
                        nt = self.standby.num_terms
                    self.current_state = DemoState(self.orchestrator, ft, nt)
                else:
                    self.current_state = self.standby
                self.current_state.enter()

            self.current_state.update_key_hold(dt)
            self.current_state.update()
            self.current_state.render(self.screen)
            self.no_borrar(dt)
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def no_borrar(self, dt):
        self._diag_timer += dt
        if not self._diag_mode:
            return

        # Dim background
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 195))
        self.screen.blit(overlay, (0, 0))

        # Heart pulse and rendering
        w, h = self.screen.get_size()
        center_x = w // 2
        center_y = h // 2
        
        # Double heart beat animation logic
        pulse = math.sin(self._diag_timer * 4.5)
        scale = 23.0 + 1.6 * (pulse * pulse)
        
        points = []
        for idx in range(150):
            t = idx * 2.0 * math.pi / 150.0
            x = 16.0 * (math.sin(t) ** 3)
            y = -(13.0 * math.cos(t) - 5.0 * math.cos(2.0*t) - 2.0 * math.cos(3.0*t) - math.cos(4.0*t))
            points.append((center_x + x * scale, center_y + y * scale + 25))

        # Draw beating crimson heart
        pygame.draw.polygon(self.screen, (220, 20, 60), points)
        pygame.draw.polygon(self.screen, (255, 100, 120), points, 4)
        pygame.draw.polygon(self.screen, (130, 10, 30), points, 2)

        # Load fonts
        try:
            font_list = ["timesnewroman", "times", "liberation serif", "dejavu serif", "serif"]
            f_reg = pygame.font.SysFont(font_list, 23)
            f_bold = pygame.font.SysFont(font_list, 25, bold=True)
        except Exception:
            f_reg = pygame.font.Font(None, 24)
            f_bold = pygame.font.Font(None, 26)

        lines = [
            ("parte de esto es gracias a ti mi esposita", f_reg),
            ("Gloria Duran Priego", f_bold),
            ("te ama tu pousi", f_reg),
            ("Gabriel Alexis Luna Gonzalez", f_bold)
        ]

        def render_outlined(text, font):
            text_surf = font.render(text, True, (255, 255, 255))
            outline_raw = font.render(text, True, (0, 0, 0))
            tw, th = text_surf.get_size()
            surf = pygame.Surface((tw + 6, th + 6), pygame.SRCALPHA)
            
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    if dx*dx + dy*dy <= 4:
                        surf.blit(outline_raw, (dx + 3, dy + 3))
            surf.blit(text_surf, (3, 3))
            return surf

        rendered_lines = [render_outlined(text, font) for text, font in lines]
        line_heights = [surf.get_height() for surf in rendered_lines]
        total_height = sum(line_heights) + (len(rendered_lines) - 1) * 12

        start_y = center_y - (total_height // 2) + 20

        current_y = start_y
        for surf in rendered_lines:
            tx = center_x - surf.get_width() // 2
            self.screen.blit(surf, (tx, current_y))
            current_y += surf.get_height() + 12

