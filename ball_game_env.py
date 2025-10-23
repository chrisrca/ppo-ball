import gymnasium as gym
import numpy as np
import math
import random
import pygame

class BallGameEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(3)  # 0: left, 1: right, 2: nothing
        low = np.array([0, 0, -20, -20, 0, -10], dtype=np.float32)
        high = np.array([400, 400, 20, 20, 400, 10], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.current_state = None
        self.screen = None
        self.max_steps = 5000
        self.steps = 0
        self.width = 400
        self.height = 400
        self.gravity = 0.5
        self.ball_radius = 8
        self.player_radius = 15
        self.player_acc = 0.8
        self.player_max_speed = 9
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        ball_pos = np.array([
            random.uniform(self.ball_radius, self.width - self.ball_radius),
            random.uniform(self.ball_radius, self.height / 2)
        ], dtype=np.float32)
        ball_vel = np.array([random.uniform(-5, 5), random.uniform(-2, 0)], dtype=np.float32)
        player_pos = np.array([
            random.uniform(self.player_radius, self.width - self.player_radius),
            self.height - 20
        ], dtype=np.float32)
        player_vel = 0.0
        self.current_state = {
            'ball_pos': ball_pos,
            'ball_vel': ball_vel,
            'player_pos': player_pos,
            'player_vel': player_vel
        }
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(), {}

    def step(self, action):
        state = self.current_state
        hit = False

        # Handle player movement
        if action == 0:  # left
            state['player_vel'] -= self.player_acc
        elif action == 1:  # right
            state['player_vel'] += self.player_acc
        elif action == 2:  # nothing
            state['player_vel'] *= 0.9
        state['player_vel'] = np.clip(state['player_vel'], -self.player_max_speed, self.player_max_speed)
        if abs(state['player_vel']) < 0.1:
            state['player_vel'] = 0
        state['player_pos'][0] += state['player_vel']
        
        # Clamp player position and stop at walls
        if state['player_pos'][0] < self.player_radius:
            state['player_pos'][0] = self.player_radius
            state['player_vel'] = 0
        elif state['player_pos'][0] > self.width - self.player_radius:
            state['player_pos'][0] = self.width - self.player_radius
            state['player_vel'] = 0

        # Update ball
        state['ball_vel'][1] += self.gravity
        state['ball_pos'] += state['ball_vel']

        # Check collision
        dx = state['ball_pos'][0] - state['player_pos'][0]
        dy = state['ball_pos'][1] - state['player_pos'][1]
        distance = math.sqrt(dx**2 + dy**2)
        if distance < self.ball_radius + self.player_radius:
            hit = True
            if distance > 0:
                nx = dx / distance
                ny = dy / distance
                overlap = (self.ball_radius + self.player_radius - distance)
                state['ball_pos'][0] += nx * overlap * 1.1
                state['ball_pos'][1] += ny * overlap * 1.1
                relative_vel_x = state['ball_vel'][0] - state['player_vel']
                relative_vel_y = state['ball_vel'][1]
                dot = relative_vel_x * nx + relative_vel_y * ny
                state['ball_vel'][0] = state['ball_vel'][0] - 2 * dot * nx
                state['ball_vel'][1] = state['ball_vel'][1] - 2 * dot * ny
                state['ball_vel'][0] *= 0.9
                state['ball_vel'][1] *= 0.9
                state['ball_vel'][0] += state['player_vel'] * 0.3
            state['ball_vel'][1] = min(state['ball_vel'][1], -8)

        # Wall bounces
        if state['ball_pos'][0] <= self.ball_radius:
            state['ball_pos'][0] = self.ball_radius
            state['ball_vel'][0] = -state['ball_vel'][0] * 0.9
        elif state['ball_pos'][0] >= self.width - self.ball_radius:
            state['ball_pos'][0] = self.width - self.ball_radius
            state['ball_vel'][0] = -state['ball_vel'][0] * 0.9

        # Ceiling bounce
        if state['ball_pos'][1] <= self.ball_radius:
            state['ball_pos'][1] = self.ball_radius
            state['ball_vel'][1] = -state['ball_vel'][1] * 0.8

        # Clamp positions
        state['ball_pos'][0] = np.clip(state['ball_pos'][0], self.ball_radius, self.width - self.ball_radius)
        state['ball_pos'][1] = max(self.ball_radius, state['ball_pos'][1])
        self.steps += 1

        # Rewards
        reward = 0
        if hit:
            reward += 10
            reward += abs(state['ball_vel'][1]) * 0.5
        terminated = state['ball_pos'][1] >= self.height - self.ball_radius
        truncated = self.steps >= self.max_steps
        if terminated:
            reward += -100
        info = {}
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        s = self.current_state
        return np.concatenate((s['ball_pos'], s['ball_vel'], s['player_pos'][0:1], [s['player_vel']]), dtype=np.float32)

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen.fill((0, 0, 0))

        # Colors
        ball_color = (255, 255, 255)
        player_color = (182, 242, 229)

        # Draw glow effect for ball
        for i in range(10):
            glow_radius = self.ball_radius + (i * 1.25)
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            alpha = int(20 * (1 - i / 10)) 
            pygame.draw.circle(glow_surface, (*ball_color, alpha), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surface, (int(self.current_state['ball_pos'][0]) - glow_radius, 
                                            int(self.current_state['ball_pos'][1]) - glow_radius))

        # Draw glow effect for player
        for i in range(10):
            glow_radius = self.player_radius + (i * 1.25)
            glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
            alpha = int(20 * (1 - i / 10)) 
            pygame.draw.circle(glow_surface, (*player_color, alpha), (glow_radius, glow_radius), glow_radius)
            self.screen.blit(glow_surface, (int(self.current_state['player_pos'][0]) - glow_radius, 
                                            int(self.current_state['player_pos'][1]) - glow_radius))

        # Draw main ball and player
        pygame.draw.circle(self.screen, ball_color, 
                          (int(self.current_state['ball_pos'][0]), int(self.current_state['ball_pos'][1])), 
                          self.ball_radius)
        pygame.draw.circle(self.screen, player_color, 
                          (int(self.current_state['player_pos'][0]), int(self.current_state['player_pos'][1])), 
                          self.player_radius)

        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None