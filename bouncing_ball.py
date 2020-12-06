#!/usr/bin/env python3

import math
import random
import time
import pygame, sys
import pygame.locals
import numpy
import threading
import queue
import signal

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

GAME_POOL = 1024
PLAYERS_PER_GENERATION = 10000
BREEDING_PER_GENERATION = 128
GAME_THREADS = 5


class Game:
    ball_x = ball_y = 0.0  # Centre
    speed = 0.01
    vector_x = 0.0  # No movement
    vector_y = 1.0  # Straight down
    paddle_x = 0.0  # Centre
    paddle_speed = 0.0  # Still
    running = False

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
                self.running = False
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
        breeding_pool = player_pool[:tobreed]
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


game_running = True


def signal_handler(sig, frame):
    global game_running
    if not game_running:
        sys.exit(1)
    game_running = False
    print("Exiting. Press Ctrl+C again to force quit")


signal.signal(signal.SIGINT, signal_handler)


generation = 1
player_pool = [Player() for _ in range(PLAYERS_PER_GENERATION)]
all_games = [Game(p) for p in player_pool]
death_pool = []

ps = [0,0,0,0,0]
top = 0
avg = 0
std = 0


game_queue = queue.Queue()
death_queue = queue.Queue()


# Game thread
def game_thread():
    thread_id = random.randint(0, 1024)
    game_pool = []
    while game_running:
        # Extend players if queue has new players available and we have space
        if len(game_pool) == 0:
            while len(game_pool) < GAME_POOL:
                try:
                    new_game = game_queue.get(block=False)
                    new_game.running = True  # New games must be started
                    game_pool.append(new_game)
                except queue.Empty:
                    if len(game_pool) == 0:
                        time.sleep(1)
                    break
        # Play the game
        for game in game_pool:
            game.tick()
            if not game.running:
                game_pool.remove(game)
                death_queue.put(game.player)


# Draw thread
def draw_thread():
    # FIXME: This may not be the most thread safe of things
    pygame.init()
    pygame.font.init()
    font = pygame.font.Font(pygame.font.get_default_font(), 16)
    BLACK = (0,0,0)
    RED = (255,0,0)
    GREEN = (0,255,0)
    WIDTH = 1280
    HEIGHT = 1024
    windowSurface = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    last_draw = 0

    text_generation = font.render(f'Generation:', True, RED)
    text_deaths = font.render(f'Players:', True, RED)
    text_fitness = font.render(f'Fit Max/Avg/Atd:', True, RED)
    text_percentile = font.render(f'Pct 99/95/90/50/25:', True, RED)


    while game_running:
        cur_time = time.time()
        if cur_time - last_draw >= 0.02:
            last_draw = cur_time
            windowSurface.fill(BLACK)
            pygame.draw.line(windowSurface, RED, (0, 0), (1000, 0))
            pygame.draw.line(windowSurface, RED, (0, 0), (0, 1000))
            pygame.draw.line(windowSurface, RED, (1000, 0), (1000, 1000))

            for i in range(len(all_games)):
                g = all_games[i]
                if not g.running:
                    continue
                col = ((i * 3263) % 256, (i * 12867) % 256, (i * 9321) % 256)
                pygame.draw.line(windowSurface, col, ((g.paddle_x - PADDLE_WIDTH + 1.0) * 500, 1000), ((g.paddle_x + PADDLE_WIDTH + 1.0) * 500, 1000))
                pygame.draw.circle(windowSurface, col, ((g.ball_x + 1.0) * 500, (g.ball_y + 1.0) * 500), 5)

            # Draw stats to screen
            text_generation_data = font.render(f'{generation}', False, RED)
            text_deaths_data = font.render(f'{len(death_pool)} / {PLAYERS_PER_GENERATION}', False, RED)
            text_fitness_data = font.render(f'{top:.03f}/{avg:.03f}/{std:.03f}', False, RED)
            text_percentile_data = font.render(f'{ps[0]:.03f}/{ps[1]:.03f}/{ps[2]:.03f}/{ps[3]:.03f}/{ps[4]:.03f}', False, RED)

            windowSurface.blit(text_generation, dest=(1010, 50))
            windowSurface.blit(text_generation_data, dest=(1150, 50))
            windowSurface.blit(text_deaths, dest=(1010, 75))
            windowSurface.blit(text_deaths_data, dest=(1100, 75))
            windowSurface.blit(text_fitness, dest=(1010, 100))
            windowSurface.blit(text_fitness_data, dest=(1020, 125))
            windowSurface.blit(text_percentile, dest=(1010, 150))
            windowSurface.blit(text_percentile_data, dest=(1020, 175))

            pygame.display.flip()
        else:
            time.sleep(0.01)


# Start the threads
threads = []
for _ in range(GAME_THREADS):
    t = threading.Thread(target=game_thread)
    t.start()
    threads.append(t)
t = threading.Thread(target=draw_thread)
t.start()
threads.append(t)


# Main thread
for game in all_games:
    game_queue.put(game)

while game_running:
    death_pool.append(death_queue.get())
    if len(death_pool) == PLAYERS_PER_GENERATION:
        death_pool = sorted(death_pool, key=lambda x: x.fitness, reverse=True)
        fitness_stats = [x.fitness for x in death_pool]
        ps = numpy.percentile(fitness_stats, q=(99, 95, 90, 50, 25))
        top = max(fitness_stats)
        avg = numpy.mean(fitness_stats)
        std = numpy.std(fitness_stats)
        print(f"End of generation {generation}. Fitness mean:{avg:0.4f} std:{std:0.4f}, 99/95/90/50/25 percentiles:", ps)
        # Iterate the generation!
        generation += 1
        player_pool = Player.cross_breed(death_pool)
        death_pool = []
        all_games = [Game(player) for player in player_pool]
        for game in all_games:
            game_queue.put(game)

for t in threads:
    t.join()
