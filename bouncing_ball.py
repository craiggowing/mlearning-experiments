#!/usr/bin/env python3

import math
import random
import time
import pygame, sys
import pygame.locals
import numpy

"""
Standard bouncing ball game where a ball will bounce around a rectangular room
with a moving paddle at the bottom.

Inputs:
        Horiz distance to ball
        Absolute distance to wall
        Ball y location
        Ball speed x
        Ball speed y
        Paddle x speed

Outputs:
* Do paddle accelerate left
* Paddle stay still
* Do paddle accelerate right

Fitness is score incremented by ball speed every hit

XXX:
You may have noticed this code is sketchy as hell and is the result of mostly
dirty hacking around to get something working, and now this is the first
generation it is actually working! (NB, it helps when you are selecting the
fittest from the population to breed and not the weakest... oops!)

TODO:
* Tidy up the code so it is actually presentable and something I would not be
  ashamed of putting forward for a code review!
* The main game engine should separate out the graphical elements and the
  game physics/AI elements so can be run independantly, ie run AI with no
  graphics, and run graphics against a replay rather than live game data
* The game ticks should be multi threaded for maximum performance.
* Each game also does not require syncronisation so for better previews it would
  be nice to slow down players as they go above the average group fitness and
  just display these games on the graphical output so it actually means something
* The graphics are a bit naff and inefficient (eg redrawing static text each frame)
  This needs jazzing up a bit!
* We have no visual indicator of the neural network other than them being dumped
  to stdout on each high score, this could get a nice graphical output
* Make Player class more generic and handle breeding as class methods instead of
  code in the game ticker
* The Player class should also handle initial random states, and loading from
  saved player pools so we can restore a generation

Further experimentation:
* The Player class should also ideally be able to mutate random hidden neurones
  and activate or deactivate neuronal connections.
* On breeding we could use double chromosomes and choose either the dominant or
  a blend of each (at random) - this in itself could be an inherited trail of the
  chromasome.
* Also porting the entire thing to use something like Tensor flow may be the end
  goal, but this is a learning activity really.
"""

PADDLE_WIDTH = 0.1  # Width in either direction (0.1 will be 1/10th of the total width)
PADDLE_ACC = 0.05

PADDLE_ACC_LEFT = 0
PADDLE_ACC_RIGHT = 1

BALL_X_SPEED = 0
BALL_Y_SPEED = 1
PADDLE_LOCATION = 2

GAME_POOL = 2048
PLAYERS_PER_GENERATION = 10000
BREEDING_PER_GENERATION = 128


class Game:
    ball_x = ball_y = 0.0  # Centre
    speed = 0.01
    vector_x = 0.0  # No movement
    vector_y = 1.0  # Straight down
    paddle_x = 0.0  # Centre
    paddle_speed = 0.0  # Still

    def __init__(self, player):
        self.player = player

    def get_outputs(self):
        return [
            (self.speed * self.vector_x),  # BALL_X_SPEED (0)
            (self.speed * self.vector_y),  # BALL_Y_SPEED (1)
            self.paddle_x,                 # PADDLE_LOCATION (2)
        ]
        
    def tick(self):
        outputs = self.get_outputs()

        player_outputs = self.player.get_outputs([
            self.paddle_x - self.ball_x, # Horiz distance to ball
            1.0 - abs(self.paddle_x),    # Absolute distance to wall
            self.ball_y,                 # Ball y location
            outputs[BALL_X_SPEED],       # Ball speed x
            outputs[BALL_Y_SPEED],       # Ball speed y
            self.paddle_speed,           # Paddle x speed
        ])

        inp_dir = player_outputs
        if inp_dir[0] >= max(inp_dir[1], inp_dir[2]):
            input_direction = -1
        elif inp_dir[2] >= max(inp_dir[0], inp_dir[1]):
            input_direction = 1
        else:
            input_direction = 0

        # Handle paddle travel
        if input_direction == -1:
            self.paddle_speed -= PADDLE_ACC
        elif input_direction == 1:
            self.paddle_speed += PADDLE_ACC
        else:  # Not moving
            self.paddle_speed = 0

        self.paddle_x += self.paddle_speed
        if self.paddle_x < -1.0:
            self.paddle_x = -1.0
            self.paddle_speed = 0
        elif self.paddle_x > 1.0:
            self.paddle_x = 1.0
            self.paddle_speed = 0

        not_dead = True

        # Handle x travel and reflection of ball
        self.ball_x += outputs[BALL_X_SPEED]
        if outputs[BALL_X_SPEED] < 0.0 and self.ball_x <= -1.0:
            self.vector_x *= -1
            self.ball_x = -self.ball_x - 2.0
        elif outputs[BALL_X_SPEED] > 0.0 and self.ball_x >= 1.0:
            self.vector_x *= -1
            self.ball_x = -self.ball_x + 2.0

        # Handle y travel and reflection of ball from roof
        self.ball_y += outputs[BALL_Y_SPEED]
        if outputs[BALL_Y_SPEED] < 0.0 and self.ball_y <= -1.0:
            self.vector_y *= -1
            self.ball_y = -self.ball_y - 2.0
        # Reflection of ball from paddle
        elif outputs[BALL_Y_SPEED] > 0.0 and self.ball_y >= 1.0:
            # Calculate the point we cross the axis
            ball_y_before_under = 1.0 - (self.ball_y - outputs[BALL_Y_SPEED])
            ball_y_over = self.ball_y - 1.0
            excess_travel_prop = ball_y_over / (ball_y_before_under + ball_y_over)
            ball_x_cross = self.ball_x - (outputs[BALL_X_SPEED] * excess_travel_prop)
            if ball_x_cross <= -1.0:
                ball_x_cross = -ball_x_cross - 2.0
            elif ball_x_cross >= 1.0:
                ball_x_cross = -ball_x_cross + 2.0

            if not (outputs[PADDLE_LOCATION] - PADDLE_WIDTH <= ball_x_cross
                    <= outputs[PADDLE_LOCATION] + PADDLE_WIDTH):
                not_dead = False
            else:
                # We hit the paddle, calculate the steepness of the reflection
                # and randomise slightly to prevent exploiting easy patterns
                # FIXME: This does not take into account the current ball angle
                # FIXME: The ball will not continue traveling from the paddle
                # until the next tick
                self.speed += 0.001
                self.player.fitness += self.speed
                reflection = (ball_x_cross - outputs[PADDLE_LOCATION]) / PADDLE_WIDTH
                reflection += (random.random() - 0.5) * 0.3
                if reflection <= -0.9:
                    reflection = -0.9
                elif reflection >= 0.9:
                    reflection = 0.9
                self.vector_x = math.sin((math.pi/2) * reflection)
                self.vector_y = -math.cos((math.pi/2) * reflection)
                self.ball_y = 1.0  # FIXME: Clamp this to the floor for now

        return not_dead


class Player:
    """
    Inputs:
        Horiz distance to ball
        Absolute distance to wall
        Ball y location
        Ball speed x
        Ball speed y
        Paddle x speed
    Hidden:
        None
    Outputs:
        Paddle move left
        Paddle stay still
        Paddle move right
    """

    inputs = 6
    hidden = 0
    outputs = 3
    fitness = 0
    _num_weights = None

    @property
    def num_weights(self):
        """
        Calcualte the number of weights we need based on inputs, hidden,
        and outputs.
        TODO: Handle hidden correctly, and groups the weights in a way which
        will allow easy breeding between instances with different neurone counts
        """
        if self._num_weights is not None:
            return self._num_weights
        self._num_weights = self.inputs * self.outputs
        return self._num_weights
        

    def __init__(self, weights=None):
        if not weights:
            weights = [(random.random() * 2) - 1.0 for _ in range(self.num_weights)]
        self.weights = weights
        # TODO: Handle hidden weights here
        # XXX: Hardcoded for now
        self.output_weights = [weights[:6], weights[6:12], weights[12:18]]

    def get_outputs(self, inputs):
        outputs = []
        for o in range(self.outputs):
            output = 0.0
            for i in range(self.inputs):
                output += inputs[i] * self.output_weights[o][i]
            outputs.append(output)
        return outputs

    def breed(self, partner, mutation_factor=0.02):
        # TODO: Handle different number of neurones/weights between individuals here
        weights = []
        for wi in range(len(self.weights)):
            if random.randint(0, 1):
                weight = self.weights[wi]
            else:
                weight = partner.weights[wi]
            weight += ((random.random() * 2.0) - 1.0) * mutation_factor
            if weight > 1.0:
                weight = 1.0
            elif weight < -1.0:
                weight = -1.0
            weights.append(weight)
        return Player(weights)

    @staticmethod
    def cross_breed(player_pool, tobreed=BREEDING_PER_GENERATION, total=PLAYERS_PER_GENERATION):
        """
        player_pool should be a list of players to breed sorted with the most
        fit at the beginning of the list.
        """
        breeding_pool = [p[1] for p in death_pool[:tobreed]]
        # Cross breed the top players against each other and apply random mutations
        offspring = []
        for p1 in range(len(breeding_pool)):
            for p2 in range(p1 + 1, len(breeding_pool)):
                offspring.append(breeding_pool[p1].breed(breeding_pool[p2]))
        # Append the parents the to pool then shorten to the required total
        # to fill in any space where new offspring were not created
        offspring.extend(player_pool)
        offspring = offspring[:total]
        return player_pool


pygame.init()
pygame.font.init()
font = pygame.font.Font(pygame.font.get_default_font(), 16)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
WIDTH = 1280
HEIGHT = 1024
windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)


generation = 1
player_pool = [Player() for _ in range(PLAYERS_PER_GENERATION)]
games = [Game(player_pool.pop()) for _ in range(GAME_POOL)]
death_pool = []

best_fitness = 0
last_draw = 0

while True:

    for i in range(GAME_POOL):
        # TODO: Multi-thread this ideally
        if games[i] and not games[i].tick():
            death_pool.append((games[i].speed, games[i].player))
            if games[i].player.fitness > best_fitness:
                best_fitness = games[i].player.fitness
                best_weights = games[i].player.weights
                print(f"New High Score! Generation {generation}, Fitness {best_fitness}")
                print(best_weights)
                print()
            # TODO: Do game cleanup here, add AI for post processing
            if player_pool:
                games[i] = Game(player_pool.pop())
            else:
                games[i] = None

    # Only redraw the screen every 1/25th of a second
    cur_time = time.time()
    if cur_time - last_draw >= 0.04:
        last_draw = cur_time
        windowSurface.fill(BLACK)
        pygame.draw.line(windowSurface, RED, (0, 0), (1000, 0))
        pygame.draw.line(windowSurface, RED, (0, 0), (0, 1000))
        pygame.draw.line(windowSurface, RED, (1000, 0), (1000, 1000))

        in_progress = 0
        for i in range(GAME_POOL):
            g = games[i]
            if not g:
                continue
            in_progress += 1
            col = ((i * 3263) % 256, (i * 12867) % 256, (i * 9321) % 256)
            pygame.draw.line(windowSurface, col, ((g.paddle_x - PADDLE_WIDTH + 1.0) * 500, 1000), ((g.paddle_x + PADDLE_WIDTH + 1.0) * 500, 1000))
            pygame.draw.circle(windowSurface, col, ((g.ball_x + 1.0) * 500, (g.ball_y + 1.0) * 500), 5)

        # Draw stats to screen
        text_generation = font.render(f'Generation: {generation}', True, RED)
        text_deaths = font.render(f'Players: {len(death_pool)} / {PLAYERS_PER_GENERATION}', True, RED)
        text_fitness = font.render(f'High Score: {best_fitness:.03f}', True, RED)
        windowSurface.blit(text_generation, dest=(1010, 50))
        windowSurface.blit(text_deaths, dest=(1010, 100))
        windowSurface.blit(text_fitness, dest=(1010, 150))

        pygame.display.flip()
    else:
        in_progress = 1

    if not in_progress:
        death_pool = sorted(death_pool, key=lambda x: x[0], reverse=True)
        percentiles = numpy.percentile([x[0] for x in death_pool], q=(99, 95, 90, 50, 25))
        avg = numpy.mean([x[0] for x in death_pool])
        std = numpy.std([x[0] for x in death_pool])
        print(f"End of generation {generation}. Fitness mean:{avg:0.4f} std:{std:0.4f}, 99/95/90/50/25 percentiles:", percentiles)
        # Iterate the generation!
        generation += 1
        player_pool = Player.cross_breed([x[1] for x in death_pool])
        death_pool = []
        games = [Game(player_pool.pop()) for _ in range(GAME_POOL)]

    

    #time.sleep(0.025)
