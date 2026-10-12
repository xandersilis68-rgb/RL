import os
import pygame
import numpy as np

# Force Pygame to run without a physical screen (Headless mode)
os.environ["SDL_VIDEODRIVER"] = "dummy"

class HeadlessCartPoleEnv:
    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # State: [cart_x, cart_velocity, pole_angle, pole_velocity]
        self.state = np.array([300.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return self.state.copy()

    def step(self, action):
        self.steps += 1
        x, x_dot, theta, theta_dot = self.state
        
        # Action: 0 = Move Left, 1 = Move Right
        force = -10.0 if action == 0 else 10.0
        
        # Simple physics simulation equations
        gravity = 9.8
        mass_cart = 1.0
        mass_pole = 0.1
        total_mass = mass_cart + mass_pole
        length = 0.5  # half pole length
        polemass_length = mass_pole * length
        
        temp = (force + polemass_length * theta_dot**2 * np.sin(theta)) / total_mass
        theta_acc = (gravity * np.sin(theta) - np.cos(theta) * temp) / (length * (4.0/3.0 - mass_pole * np.cos(theta)**2 / total_mass))
        x_acc = temp - polemass_length * theta_acc * np.cos(theta) / total_mass
        
        # Update states (Euler integration)
        dt = 0.02
        x += dt * x_dot
        x_dot += dt * x_acc
        theta += dt * theta_dot
        theta_dot += dt * theta_acc
        
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        
        # Check failure conditions
        done = bool(
            x < 0 or x > self.width or
            theta < -0.4 or theta > 0.4 or
            self.steps >= 500
        )
        
        reward = 1.0 if not done else 0.0
        return self._get_obs(), reward, done

    def render_frame(self):
        """Draws the game into memory and extracts the image array."""
        self.screen.fill((255, 255, 255)) # White background
        x, _, theta, _ = self.state
        
        # Draw Cart (Black rectangle)
        cart_w, cart_h = 80, 40
        cart_y = 300
        pygame.draw.rect(self.screen, (0, 0, 0), (int(x - cart_w/2), cart_y, cart_w, cart_h))
        
        # Draw Pole (Red line)
        pole_len = 100
        pole_end_x = x + pole_len * np.sin(theta)
        pole_end_y = cart_y - pole_len * np.cos(theta)
        pygame.draw.line(self.screen, (255, 0, 0), (int(x), cart_y), (int(pole_end_x), int(pole_end_y)), 6)
        
        # Convert the Pygame surface directly into an RGB pixel array
        frame = pygame.surfarray.array3d(self.screen)
        return np.transpose(frame, (1, 0, 2)) # Fix axis alignment for video writing
