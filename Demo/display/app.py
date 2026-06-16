import pygame
import sys
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

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if type(self.current_state).__name__ == "StandbyState":
                        self.running = False
                    else:
                        self.current_state.done = True
                        self.current_state.next_state = "standby"
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F4:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_ALT:
                        self.running = False
                elif event.type == pygame.KEYDOWN:
                    self.current_state.on_key_down(event.key)
                    self.current_state.handle_event(event)
                elif event.type == pygame.KEYUP:
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
            pygame.display.flip()

        pygame.quit()
        sys.exit()
