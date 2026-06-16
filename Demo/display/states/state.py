from abc import ABC, abstractmethod
import pygame

class AppState(ABC):
    def __init__(self):
        self.done = False
        self.next_state = None
        self.key_held = {}
        self.key_held_timer = {}

    @abstractmethod
    def handle_event(self, event):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def render(self, surface):
        pass

    def enter(self):
        self.done = False
        self.next_state = None
        self.key_held = {}
        self.key_held_timer = {}

    def exit(self):
        pass

    def on_key_down(self, key):
        self.key_held[key] = True
        self.key_held_timer[key] = 0.0

    def on_key_up(self, key):
        if key in self.key_held:
            del self.key_held[key]
        if key in self.key_held_timer:
            del self.key_held_timer[key]

    def update_key_hold(self, dt):
        for key in list(self.key_held.keys()):
            self.key_held_timer[key] = self.key_held_timer.get(key, 0.0) + dt
        return self.key_held, self.key_held_timer
