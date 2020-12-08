#!/usr/bin/env python3

import math
import random
import time
import pygame, sys
import pygame.locals
import numpy
from multiprocessing import Process, Queue, Value
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
GAME_THREADS = 10

game_running = Value('i', 1)


def signal_handler_main(sig, frame):
    global game_running
    if not game_running.value:
        sys.exit(1)
    game_running.value = 0
    print("Exiting. Press Ctrl+C again to force quit")


quit_signal = False


def signal_handler_thread(sig, frame):
    global quit_signal
    if quit_signal:
        sys.exit(1)
    print("Waiting for main process to quit. Press Ctrl+C again to force quit")
    quit_signal = True


def sigmoid(x):
    """
    Generic sigmoid curve between y=0 and 1, passing through 0,0.5
    """
    return 1 / (1 + math.exp(-x))


def trans_sigmoid(x):
    """
    Sigmoid curve translated to pass through 0,0 and gives a y value
    between -1 and 1
    """
    return 2 * (sigmoid(x) - 0.5)


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
    num_weights = 18  # 6*3 connections from in to out
    num_biases = 3  # 3 output neurones

    def __init__(self, weights=None, biases=None):
        if not weights:
            weights = [(random.random() * 2) - 1.0 for _ in range(self.num_weights)]
        if not biases:
            biases = [(random.random() * 2) - 1.0 for _ in range(self.num_biases)]
        self.biases = biases
        self.weights = weights
        # TODO: Handle hidden weights here
        # XXX: Hardcoded for now
        self.output_weights = [weights[:6], weights[6:12], weights[12:18]]
        self.output_bias = biases

    def get_outputs(self, inputs):
        outputs = []
        for o in range(self.outputs):
            output = 0.0
            for i in range(self.inputs):
                output += trans_sigmoid(inputs[i]) * self.output_weights[o][i]
            outputs.append(sigmoid(output + self.output_bias[o]))
        return outputs

    def breed(self, partner, mutation_factor=0.05, mutation_prop=0.2):
        # TODO: Handle different number of neurones/weights between individuals here
        weights = []
        biases = []
        crossover_point = random.randint(0, len(self.weights) + 1)
        for wi in range(len(self.weights)):
            if wi < crossover_point:
                weight = self.weights[wi]
            else:
                weight = partner.weights[wi]
            if random.random() <= mutation_prop:
                weight += ((random.random() * 2.0) - 1.0) * mutation_factor
            if weight > 1.0:
                weight = 1.0
            elif weight < -1.0:
                weight = -1.0
            weights.append(weight)
        for wi in range(len(self.biases)):
            if wi < (crossover_point // 3):
                bias = self.biases[wi]
            else:
                bias = partner.biases[wi]
            if random.random() <= mutation_prop:
                bias += ((random.random() * 2.0) - 1.0) * mutation_factor
            if bias > 1.0:
                bias = 1.0
            elif bias < -1.0:
                bias = -1.0
            biases.append(bias)
        return Player(weights, biases)

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


generation = 1

ps = [0,0,0,0,0]
top = 0
avg = 0
std = 0

stats = {
    'generation': generation,
    'top': top,
    'avg': avg,
    'std': std,
    'ps': ps,
    'death_pool': 0,
}


game_queue = Queue()
death_queue = Queue()


# Game thread
def game_thread(game_queue, death_queue, game_running):
    signal.signal(signal.SIGINT, signal_handler_thread)
    game_pool = []
    death_pool = []
    while game_running.value:
        # Extend players if queue has new players available and we have space
        # Batch read from game_pool and write to death_queue to avoid threads stalling
        if len(game_pool) == 0:
            for player in death_pool:
                death_pool.remove(player)
                death_queue.put(player)
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
                death_pool.append(game.player)


# Draw engine
class DrawWorker:
    BLACK = (0,0,0)
    RED = (255,0,0)
    GREEN = (0,255,0)
    WIDTH = 1280
    HEIGHT = 1024

    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.windowSurface = pygame.display.set_mode((DrawWorker.WIDTH, DrawWorker.HEIGHT), 0, 32)
        self.last_draw = 0

        self.text_generation = self.font.render(f'Generation:', True, DrawWorker.RED)
        self.text_deaths = self.font.render(f'Players:', True, DrawWorker.RED)
        self.text_fitness = self.font.render(f'Fit Max/Avg/Std:', True, DrawWorker.RED)
        self.text_percentile = self.font.render(f'Pct 99/95/90/50/25:', True, DrawWorker.RED)

    def draw_frame(self, games, stats, frame_delay=0.2):
        cur_time = time.time()
        if not frame_delay or (cur_time - self.last_draw >= frame_delay):
            self.last_draw = cur_time
            self.windowSurface.fill(DrawWorker.BLACK)
            pygame.draw.line(self.windowSurface, DrawWorker.RED, (0, 0), (1000, 0))
            pygame.draw.line(self.windowSurface, DrawWorker.RED, (0, 0), (0, 1000))
            pygame.draw.line(self.windowSurface, DrawWorker.RED, (1000, 0), (1000, 1000))

            for i in range(len(games)):
                g = games[i]
                col = ((i * 3263) % 256, (i * 12867) % 256, (i * 9321) % 256)
                pygame.draw.line(self.windowSurface, col, ((g.paddle_x - PADDLE_WIDTH + 1.0) * 500, 1000), ((g.paddle_x + PADDLE_WIDTH + 1.0) * 500, 1000))
                pygame.draw.circle(self.windowSurface, col, ((g.ball_x + 1.0) * 500, (g.ball_y + 1.0) * 500), 5)

            # Draw stats to screen
            text_generation_data = self.font.render(f'{stats["generation"]}',
                                                    True, DrawWorker.RED)
            text_deaths_data = self.font.render(f'{stats["death_pool"]} / {PLAYERS_PER_GENERATION}',
                                                True, DrawWorker.RED)
            text_fitness_data = self.font.render(f'{stats["top"]:.03f}/{stats["avg"]:.03f}/{stats["std"]:.03f}',
                                                True, DrawWorker.RED)
            text_percentile_data = self.font.render(
                f'{stats["ps"][0]:.03f}/{stats["ps"][1]:.03f}/'
                f'{stats["ps"][2]:.03f}/{stats["ps"][3]:.03f}/'
                f'{stats["ps"][4]:.03f}', True, DrawWorker.RED)

            self.windowSurface.blit(self.text_generation, dest=(1010, 50))
            self.windowSurface.blit(text_generation_data, dest=(1150, 50))
            self.windowSurface.blit(self.text_deaths, dest=(1010, 75))
            self.windowSurface.blit(text_deaths_data, dest=(1100, 75))
            self.windowSurface.blit(self.text_fitness, dest=(1010, 100))
            self.windowSurface.blit(text_fitness_data, dest=(1020, 125))
            self.windowSurface.blit(self.text_percentile, dest=(1010, 150))
            self.windowSurface.blit(text_percentile_data, dest=(1020, 175))

            pygame.display.flip()
        else:
            time.sleep(0.01)


# Start the threads
threads = []
for _ in range(GAME_THREADS):
    t = Process(target=game_thread, args=(game_queue, death_queue, game_running))
    t.start()
    threads.append(t)


# Main thread
player_pool = [Player() for _ in range(PLAYERS_PER_GENERATION)]
all_games = [Game(p) for p in player_pool]
death_pool = []

for game in all_games:
    game_queue.put(game)

# Only the main process should handle signals
signal.signal(signal.SIGINT, signal_handler_main)

drawer = DrawWorker()
draw_game_pool = []
draw_death_pool = []
while game_running.value:
    # Collect dead players
    while True:
        try:
            death_pool.append(death_queue.get(block=False))
        except queue.Empty:
            break
    # Handle dead players
    if len(death_pool) == PLAYERS_PER_GENERATION:
        death_pool = sorted(death_pool, key=lambda x: x.fitness, reverse=True)
        fitness_stats = [x.fitness for x in death_pool]
        ps = numpy.percentile(fitness_stats, q=(99, 95, 90, 50, 25))
        top = max(fitness_stats)
        avg = numpy.mean(fitness_stats)
        std = numpy.std(fitness_stats)
        print(f"End of generation {generation}. Fitness top:{top:0.4f} mean:{avg:0.4f} std:{std:0.4f}, 99/95/90/50/25 percentiles:", ps)
        # Iterate the generation!
        generation += 1
        player_pool = Player.cross_breed(death_pool)
        death_pool = []
        all_games = [Game(player) for player in player_pool]
        for game in all_games:
            game_queue.put(game)
        stats = {
            'generation': generation,
            'top': top,
            'avg': avg,
            'std': std,
            'ps': ps,
        }
    # Draw select few games and print stats
    stats["death_pool"] = len(death_pool)
    if len(draw_game_pool) < 10:
        for player in draw_death_pool:
            draw_death_pool.remove(player)
            death_queue.put(player)
        while len(draw_game_pool) < 10:
            try:
                new_game = game_queue.get(block=False)
                new_game.running = True  # New games must be started
                draw_game_pool.append(new_game)
            except queue.Empty:
                break
    for game in draw_game_pool:
        game.tick()
        if not game.running:
            draw_game_pool.remove(game)
            draw_death_pool.append(game.player)
    drawer.draw_frame(draw_game_pool, stats, frame_delay=0)

print("Main process exiting, waiting for children...")
for t in threads:
    t.join()
game_queue.close()
game_queue.join_thread()
death_queue.close()
death_queue.join_thread()
print("All children terminated, main process exiting.")
sys.exit(0)
