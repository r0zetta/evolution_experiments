from ursina import *
import random, sys, time, os, json, re, pickle, operator, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from scipy.spatial import distance

STILL = 0
UP = 1
UP_RIGHT = 2
RIGHT = 3
DOWN_RIGHT = 4
DOWN = 5
DOWN_LEFT = 6
LEFT = 7
UP_LEFT = 8
DIRECTIONS = [UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, LEFT, UP_LEFT]

class Net(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, fc1_weights, fc2_weights):
        super(Net, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.state_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.action_size, bias=False)
        self.fc1.weight.data =  fc1_weights
        self.fc2.weight.data =  fc2_weights

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(F.relu(self.fc2(x)))
        return x

    def get_action(self, state):
        with torch.no_grad():
            state = state.float()
            ret = self.forward(Variable(state))
            action = torch.argmax(ret)
            return action

class GN_model:
    def __init__(self, state_size, action_size, hidden_size, fc1_weights, fc2_weights):
        self.policy = Net(state_size, action_size, hidden_size, fc1_weights, fc2_weights)

    def get_action(self, state):
        action = self.policy.get_action(state)
        return action


class Agent:
    def __init__(self, x, y, state_size, action_size, hidden_size, genome, gi):
        self.xpos = x
        self.ypos = y
        self.speed = 1
        self.fitness = 0
        self.gi = gi
        fc1_weights = torch.Tensor(np.reshape(genome[:state_size*hidden_size],
                                              (hidden_size, state_size)))
        fc2_weights = torch.Tensor(np.reshape(genome[state_size*hidden_size:],
                                              (action_size, hidden_size)))
        self.model = GN_model(state_size, action_size, hidden_size, fc1_weights, fc2_weights)
        self.state = None
        self.episode_steps = 0
        self.last_success = None
        self.direction = 0
        self.entity = None

    def get_action(self):
        return self.model.get_action(self.state)

class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = 0
        self.speed = 0
        self.friction = 0.95
        self.hit_speed = 2
        self.hit_cooldown = 0
        self.entity = None

class game_space:
    def __init__(self, width, height, team_size=5,
                 max_episode_len=200, scaling=10, visuals=False, savedir="save"):
        self.steps = 0
        self.scaling = scaling
        self.visuals = visuals
        self.savedir = savedir
        self.max_episode_len = max_episode_len
        self.width = width
        self.height = height
        self.bottom = int((self.height*self.scaling)/2) - self.scaling
        self.top = -1 * self.bottom
        self.right = int((self.width*self.scaling)/2) - self.scaling
        self.left = -1 * self.right
        print(self.top, self.bottom, self.left, self.right)
        self.action_size = 9
        self.hidden_size = 16
        self.pool_size = 1000
        self.state_size = self.get_state_size()
        self.genome_size = (self.state_size*self.hidden_size) + (self.action_size*self.hidden_size)
        self.agent_types = ["pigs", "sheep"]
        self.textures = {"pigs":"hog", "sheep":"yak"}
        self.num_agents = {}
        self.num_agents["pigs"] = team_size
        self.num_agents["sheep"] = team_size
        self.ball = None
        self.previous_hit = None
        self.genome_pool = {}
        self.best_policies = {}
        for t in self.agent_types:
            self.best_policies[t] = []
            self.genome_pool[t] = []
            if os.path.exists(savedir + "/" + t + "_best_policies.pkl"):
                print("Loading best " + t + " policies.")
                self.best_policies[t] = self.load_best_policies(t)
            if os.path.exists(savedir + "/" + t + "_genome_pool.pkl"):
                print("Loading " + t + " genomes.")
                self.genome_pool[t] = self.load_genomes(t)
            else:
                print("Creating " + t + " genomes.")
                self.genome_pool[t] = self.make_genomes()
        self.reset()

    def reset(self):
        self.ball = Ball(0, 0)
        self.agents = {}
        for t in self.agent_types:
            self.agents[t] = []
            for index in range(self.num_agents[t]):
                self.agents[t].append(None)
        self.game_space = self.make_empty_game_space()
        for t in self.agent_types:
            for index, agent in enumerate(self.agents[t]):
                self.create_new_agent(index, t)
        for t in self.agent_types:
            for index, agent in enumerate(self.agents[t]):
                state = self.get_agent_state(index, t)
                self.agents[t][index].state = state

    def step(self):
        for t in self.agent_types:
            for index in range(len(self.agents[t])):
                action = gs.agents[t][index].get_action()
                self.move_agent(index, action, t)
        prev_hit_team = None
        prev_hit_player = None
        if self.previous_hit is not None:
            prev_hit_team, prev_hit_player = self.previous_hit

        who_hit = self.move_ball()
        if len(who_hit) > 0:
            for item in who_hit:
                t, p = item
                # Get fitness for interacting with the ball
                self.agents[t][p].fitness += 1
                if t == prev_hit_team:
                    if p != prev_hit_player:
                        # If the ball is passed to a player on the same team
                        # add to fitness of play who made the pass
                        self.agents[prev_hit_team][prev_hit_player].fitness += 1
                self.previous_hit = [t, p]
        self.steps+=1

    def abs_to_grid(self, absx, absy):
        nx = int(absx/self.scaling) + int(self.width/2)
        ny = int(absy/self.scaling) + int(self.height/2)
        return nx, ny

    def grid_to_abs(self, gridx, gridy):
        x = self.scaling * (gridx-int(self.width/2))
        y = self.scaling * (gridy-int(self.height/2))
        return x, y

    def make_empty_game_space(self):
        space = np.zeros((self.height, self.width), dtype=int)
        return space

    def create_new_agent(self, index, atype):
        item = self.get_random_empty_space(1)
        xabs, yabs = item[0]
        state_size = self.get_state_size()
        action_size = self.action_size

        selectable = []
        for i, item in enumerate(self.genome_pool[atype]):
            g, f = item
            if f is None:
                selectable.append(i)
        gi = random.choice(selectable)
        item = self.genome_pool[atype][gi]
        genome, fitness = item

        self.agents[atype][index] = Agent(xabs, yabs, state_size, action_size,
                                          self.hidden_size, genome, gi)

    def respawn_agent(self, index, atype):
        old_entity = self.agents[atype][index].entity
        gi = self.agents[atype][index].gi
        item = self.genome_pool[atype][gi]
        genome, fitness = item
        fitness = self.agents[atype][index].fitness
        self.genome_pool[atype][gi] = [genome, fitness]
        self.create_new_agent(index, atype)
        state = self.get_agent_state(index, atype)
        self.agents[atype][index].state = state
        if old_entity is not None:
            self.agents[atype][index].entity = old_entity
            xabs = self.agents[atype][index].xpos
            yabs = self.agents[atype][index].ypos
            self.agents[atype][index].entity.position = (xabs, yabs, (-1*self.scaling))

    def get_random_empty_space(self, num):
        coords = []
        for _ in range(num):
            xpos = random.randint(self.left, self.right)
            ypos = random.randint(self.top, self.bottom)
            coords.append([ypos, xpos])
        return coords


    def ball_collision(self):
        hit_directions = []
        who_hit = []
        # Check for collision with sides of pitch
        if self.ball.y <= self.top:
            if self.ball.direction == UP:
                hit_directions.append(DOWN)
            elif self.ball.direction == UP_RIGHT:
                hit_directions.append(DOWN_RIGHT)
            elif self.ball.direction == UP_LEFT:
                hit_directions.append(DOWN_LEFT)
        elif self.ball.y >= self.bottom:
            if self.ball.direction == DOWN:
                hit_directions.append(UP)
            elif self.ball.direction == DOWN_RIGHT:
                hit_directions.append(UP_RIGHT)
            elif self.ball.direction == DOWN_LEFT:
                hit_directions.append(UP_LEFT)
        if self.ball.x <= self.left:
            if self.ball.direction == LEFT:
                hit_directions.append(RIGHT)
            elif self.ball.direction == UP_LEFT:
                hit_directions.append(UP_RIGHT)
            elif self.ball.direction == DOWN_LEFT:
                hit_directions.append(DOWN_RIGHT)
        elif self.ball.x >= self.right:
            if self.ball.direction == RIGHT:
                hit_directions.append(LEFT)
            elif self.ball.direction == DOWN_RIGHT:
                hit_directions.append(DOWN_LEFT)
            elif self.ball.direction == UP_RIGHT:
                hit_directions.append(UP_LEFT)
        # Check for collisions with players
        lt = -1 * self.scaling
        lt1 = -1 * int(self.scaling/2)
        ut = self.scaling
        ut1 = int(self.scaling/2)
        for t in self.agent_types:
            for index in range(len(self.agents[t])):
                dx = self.agents[t][index].xpos
                dy = self.agents[t][index].ypos
                ox = dx - self.ball.x
                oy = dy - self.ball.y
                hd = None
                if (ox > lt and ox < ut) and (oy > lt and oy < ut):
                    if oy < lt1 and oy > lt:
                        if ox <= lt1 and ox > lt:
                            hd = DOWN_RIGHT
                        elif ox >= ut1 and ox < ut:
                            hd = DOWN_LEFT
                        else:
                            hd = DOWN
                    elif oy >= ut1 and oy < ut:
                        if ox <= lt1 and ox > lt:
                            hd = UP_RIGHT
                        elif ox >= ut1 and ox < ut:
                            hd = UP_LEFT
                        else:
                            hd = UP
                    elif ox <= lt1 and ox > lt:
                        if oy <= lt1 and oy > lt:
                            hd = DOWN_RIGHT
                        elif oy >= ut1 and oy < ut:
                            hd = UP_RIGHT
                        else:
                            hd = RIGHT
                    elif ox >= ut1 and ox < ut:
                        if oy <= lt1 and oy > lt:
                            hd = DOWN_LEFT
                        elif oy >= ut1 and oy < ut:
                            hd = UP_LEFT
                        else:
                            hd = LEFT
                if hd is not None:
                    hit_directions.append(hd)
                    who_hit.append([t, index])
        return hit_directions, who_hit

    def move_ball(self):
        # See if it collided with anything
        who_hit = []
        if self.ball.hit_cooldown == 0:
            hit_directions, who_hit = self.ball_collision()
            if len(hit_directions) > 0:
                self.ball.direction = random.choice(hit_directions)
                if len(who_hit) > 0:
                    self.ball.speed = self.ball.hit_speed
                self.ball.hit_cooldown = 5
        # Move ball
        self.ball.x, self.ball.y = self.move_item(self.ball.x,
                                                  self.ball.y,
                                                  self.ball.direction,
                                                  self.ball.speed)
        self.ball.speed = self.ball.speed * self.ball.friction
        if self.ball.entity is not None:
            self.ball.entity.position = (self.ball.x, self.ball.y, (-1*self.scaling))
        self.ball.hit_cooldown -= 1
        if self.ball.hit_cooldown < 0:
            self.ball.hit_cooldown = 0
        return who_hit

    def get_direction_mods(self, direction):
        modx = 0
        mody = 0
        if direction == UP:
            mody = -1
        elif direction == UP_RIGHT:
            mody = -1
            modx = 1
        elif direction == RIGHT:
            modx = 1
        elif direction == DOWN_RIGHT:
            mody = 1
            modx = 1
        elif direction == DOWN:
            mody = 1
        elif direction == DOWN_LEFT:
            mody = 1
            modx = -1
        elif direction == LEFT:
            modx = -1
        elif direction == UP_LEFT:
            mody = -1
            modx = -1
        return modx, mody

    def move_item(self, x, y, direction, speed):
        nx = x
        ny = y
        modx, mody = self.get_direction_mods(direction)
        ny = ny + (mody*speed)
        nx = nx + (modx*speed)
        if nx < self.left:
            nx = self.left
        if nx > self.right:
            nx = self.right
        if ny > self.bottom:
            ny = self.bottom
        if ny < self.top:
            ny = self.top
        return nx, ny

    def move_agent(self, index, action, atype):
        if action == 0: # do nothing
            pass
        else:
            xabs = self.agents[atype][index].xpos
            yabs = self.agents[atype][index].ypos
            speed = self.agents[atype][index].speed
            nx, ny = self.move_item(xabs, yabs, action, speed)
            self.agents[atype][index].xpos = nx
            self.agents[atype][index].ypos = ny
            if self.agents[atype][index].entity is not None:
                self.agents[atype][index].entity.position = (nx, ny, (-1*self.scaling))
        self.agents[atype][index].episode_steps += 1
        if self.agents[atype][index].episode_steps > self.max_episode_len:
            self.respawn_agent(index, atype)

    def distance(self, xa, ya, xb, yb):
        dst = distance.euclidean([xa, ya], [xb, yb])
        return dst

    def get_directions(self, xpos, ypos, positions, num):
        ind = {}
        for index, item in enumerate(positions):
            ey, ex = item
            dist = self.distance(ex, ey, xpos, ypos)
            ind[index] = dist
        nearest_indices = []
        for index, dist in sorted(ind.items(), key=operator.itemgetter(1),reverse=False):
            nearest_indices.append(index)
            if len(nearest_indices) >= num:
                break
        directions = []
        for index in nearest_indices:
            item = positions[index]
            ey, ex = item
            xoff = ex-xpos
            if xoff > 0:
                xoff = 1
            if xoff < 0:
                xoff = -1
            yoff = ey-ypos
            if yoff > 0:
                yoff = 1
            if yoff < 0:
                yoff = -1
            directions.append([xoff,yoff])
        return directions

    def get_ball_direction(self, xpos, ypos):
        positions = [[self.ball.y, self.ball.x]]
        return self.get_directions(xpos, ypos, positions, 1)

    def get_team_directions(self, xpos, ypos, atype, num):
        positions = []
        for agent in self.agents[atype]:
            positions.append([agent.ypos, agent.xpos])
        return self.get_directions(xpos, ypos, positions, num)

    def add_directions_to_state(self, state, directions):
        if directions is not None:
            for item in directions:
                yp, xp = item
                state.append(xp)
                state.append(yp)
        return state

    def get_state_size(self):
        state_size = 13
        return state_size

    def get_agent_state(self, index, atype):
        ot = "pigs"
        if atype == "pigs":
            ot = "sheep"
        xabs = self.agents[atype][index].xpos
        yabs = self.agents[atype][index].ypos
        modx, mody = self.get_direction_mods(self.ball.direction)
        state = [modx, mody, self.ball.speed]
        # Get direction of ball
        self.add_directions_to_state(state, self.get_ball_direction(xabs, yabs))
        # Get directions of 4 closest team mates
        self.add_directions_to_state(state, self.get_team_directions(xabs, yabs, atype, 2))
        # Get directions of 4 closest opposing team players
        self.add_directions_to_state(state, self.get_team_directions(xabs, yabs, ot, 2))
        state = np.array(state)
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0)
        return state

    def make_genomes(self):
        genome_pool = []
        for _ in range(self.pool_size):
            genome = np.random.uniform(-1, 1, self.genome_size)
            fitness = None
            genome_pool.append([genome, fitness])
        return genome_pool

    def reproduce_genome(self, g1, g2, num_offspring):
        new_genomes = []
        for _ in range(num_offspring):
            s = random.randint(10, len(g1)-10)
            g3 = np.concatenate((g1[:s], g2[s:]))
            new_genomes.append(g3)
            g4 = np.concatenate((g2[:s], g1[s:]))
            new_genomes.append(g4)
        return new_genomes

    def mutate_genome(self, g, num_mutations):
        new_genomes = []
        for _ in range(num_mutations):
            n = int(0.05 * len(g))
            indices = random.sample(range(len(g)), n)
            gm = g
            for index in indices:
                val = random.uniform(-1, 1)
                gm[index] = val
            new_genomes.append(gm)
        return new_genomes

    def get_min_best_fitness(self, atype):
        best_len = len(self.best_policies[atype])
        if best_len < self.pool_size:
            return 0
        min_val = None
        min_index = None
        for index, item in enumerate(self.best_policies[atype]):
            genome, fitness = item
            if min_val is None:
                min_val = fitness
                min_index = index
            if fitness < min_val:
                min_val = fitness
                min_index = index
        return min_val, min_index

    def replace_best_fitness_entry(self, atype, index, genome, fitness):
        self.best_policies[atype][index] = ([genome, fitness])

    def trim_best_policies(self, atype):
        bpl = len(self.best_policies[atype])
        if bpl > self.pool_size:
            while len(self.best_policies[atype]) > self.pool_size:
                val, index = self.get_min_best_fitness(atype)
                del self.best_policies[atype][index]

    def add_best_fitness_entry(self, atype, genome, fitness):
        self.trim_best_policies(atype)
        if fitness < 1:
            return
        best_len = len(self.best_policies[atype])
        if best_len >= self.pool_size:
            mv, mi = self.get_min_best_fitness(atype)
            if fitness > mv:
                self.replace_best_fitness_entry(atype, mi, genome, fitness)
        else:
            self.best_policies[atype].append([genome, fitness])

    def get_best_policy_stats(self, atype):
        f = []
        l = len(self.best_policies[atype])
        for item in self.best_policies[atype]:
            genome, fitness = item
            f.append(fitness)
        if len(f) > 0:
            return np.mean(f), l
        else:
            return 0, 0

    def get_best_policies(self, atype, num):
        vi = {}
        for index, item in enumerate(self.best_policies[atype]):
            genome, fitness = item
            vi[index] = fitness
        ret = []
        for index, fitness in sorted(vi.items(), key=operator.itemgetter(1),reverse=True):
            genome = self.best_policies[atype][index][0]
            ret.append(genome)
            if len(ret) >= num:
                break
        return ret

    def get_best_genomes(self, atype, threshold):
        pool = self.genome_pool[atype]
        fit_genomes = []
        vi = {}
        for index, item in enumerate(pool):
            genome, fitness = item
            if fitness is not None:
                if fitness > 0:
                    vi[index] = fitness
        count = 0
        for index, fitness in sorted(vi.items(), key=operator.itemgetter(1),reverse=True):
            genome = pool[index][0]
            fit_genomes.append(genome)
            self.add_best_fitness_entry(atype, genome, fitness)
            count += 1
            if count > int(len(vi) * threshold):
                break
        return fit_genomes

    def create_new_genome_pool(self, atype):
        # Get genomes from previous pool that had positive fitness
        new_genomes = []
        msg = ""
        pool = self.genome_pool[atype]
        # Grab unused genomes
        for item in pool:
            genome, fitness = item
            if fitness is None:
                new_genomes.append(genome)
        threshold = 0.30
        fit_genomes = self.get_best_genomes(atype, threshold)
        best_policies = []
        if len(self.best_policies[atype]) > 20:
            best_policies = self.get_best_policies(atype, 20)
        msg += atype + ": Previous pool had " + str(len(fit_genomes)) + " fit genomes.\n"
        mutated_fit = []
        for item in fit_genomes:
            mutated_fit.extend(self.mutate_genome(item, 4))
        for item in best_policies:
            mutated_fit.extend(self.mutate_genome(item, 4))
        msg += "New genomes from mutations: " + str(len(mutated_fit)) + "\n"
        # Select pairs to reproduce and mutate
        repr_genomes = []
        if len(fit_genomes) > 2:
            num_pairs = min(int(self.pool_size/50), int(len(fit_genomes)))
            for _ in range(num_pairs):
                g1, g2 = random.sample(fit_genomes, 2)
                offspring = (self.reproduce_genome(g1, g2, 4))
                repr_genomes.extend(offspring)
                for item in offspring:
                    repr_genomes.extend(self.mutate_genome(item, 4))
        msg += "New genomes from reproduction: " + str(len(repr_genomes)) + "\n"
        new_genomes.extend(fit_genomes)
        new_genomes.extend(best_policies)
        new_genomes.extend(mutated_fit)
        new_genomes.extend(repr_genomes)
        msg += "New genome pool size: " + str(len(new_genomes)) + "\n"
        if len(new_genomes) < self.pool_size:
            pad = self.pool_size - len(new_genomes)
            msg += "Adding " + str(pad) + " new random genomes." + "\n"
            for _ in range(pad):
                genome = np.random.uniform(-1, 1, self.genome_size)
                new_genomes.append(genome)
        elif len(new_genomes) > self.pool_size:
            msg += "Taking a random sample from new genome pool." + "\n"
            new_genomes = random.sample(new_genomes, self.pool_size)
        genome_pool = []
        for g in new_genomes:
            genome_pool.append([np.array(g), None])
        self.genome_pool[atype] = genome_pool
        msg += "\n"
        return msg

    def get_genome_statistics(self, atype):
        success = 0
        unused = 0
        pool = self.genome_pool[atype]
        for item in pool:
            genome, fitness = item
            if fitness is not None:
                if fitness > 0:
                    success += 1
            else:
                unused += 1
        return success, unused

    def get_genome_fitness(self, atype):
        f = []
        pool = self.genome_pool[atype]
        for item in pool:
            genome, fitness = item
            if fitness is not None:
                f.append(fitness)
        if len(f) > 0:
            return np.mean(f), max(f)
        else:
            return 0,0

    def save_best_policies(self, atype):
        with open(self.savedir + "/" + atype + "_best_policies.pkl", "wb") as f:
            f.write(pickle.dumps(self.best_policies[atype]))

    def load_best_policies(self, atype):
        n = []
        with open(self.savedir + "/" + atype + "_best_policies.pkl", "rb") as f:
            n = pickle.load(f)
        return n

    def save_genomes(self, atype):
        with open(self.savedir + "/" + atype + "_genome_pool.pkl", "wb") as f:
            f.write(pickle.dumps(self.genome_pool[atype]))

    def load_genomes(self, atype):
        n = []
        with open(self.savedir + "/" + atype + "_genome_pool.pkl", "rb") as f:
            n = pickle.load(f)
        return n

# update callback for ursina
def update():
    gs.step()
    direction = gs.ball.direction
    speed = gs.ball.speed
    rotx = 0
    roty = 0
    if direction == UP:
        roty = 100 * speed
    elif direction == UP_RIGHT:
        roty = 100 * speed
        rotx = 100 * speed
    elif direction == RIGHT:
        rotx = 100 * speed
    elif direction == DOWN_RIGHT:
        roty = -100 * speed
    elif direction == DOWN:
        roty = -100 * speed
    elif direction == DOWN_LEFT:
        roty = -100 * speed
        rotx = 100 * speed
    elif direction == LEFT:
        rotx = 100 * speed
    elif direction == UP_LEFT:
        roty = 100 * speed
        rotx = 100 * speed
    gs.ball.entity.rotation = (rotx, roty, 0)

# Train the game
def msg(gs):
    colors = {"pigs":"32;40", "sheep":"35;40"}
    msg = "Steps: " + str(gs.steps) + " Episode length: " + str(gs.max_episode_len) 
    msg += " Genome size: " + str(gs.genome_size) 
    msg += " Pool size: " + str(gs.pool_size) + "\n\n"
    for t in gs.agent_types:
        s, u = gs.get_genome_statistics(t)
        f, m = gs.get_genome_fitness(t)
        pf, pn = gs.get_best_policy_stats(t)
        msg += "\x1b[1;"+colors[t]+"m"+t+"\x1b[0m"
        msg += ": Success: " + str(s) + " Unused: " + str(u) 
        msg += " Fitness: " + "%.2f"%f + " Max: " + str(m) + "\n"
        msg += "Best policy fitness: " + "%.2f"%pf + " Num: " + str(pn) + "\n"
        msg += "[ "
        p = prev_stats[t]
        for s in p[-10:]:
            msg += "%.2f"%s + " "
        msg += "]\n\n"
    return msg

random.seed(1337)

print_visuals = False
scaling = 5
game_space_width = 25
game_space_height = 15
team_size = 3
max_episode_len = 50 * scaling
savedir = "footie_save"
if not os.path.exists(savedir):
    os.makedirs(savedir)

gs = game_space(game_space_width,
                game_space_height,
                team_size=team_size,
                max_episode_len=max_episode_len,
                scaling=scaling,
                visuals=print_visuals,
                savedir=savedir)

prev_stats = {}
for t in gs.agent_types:
    prev_stats[t] = []
    if os.path.exists(savedir + "/evolution_stats_"+t+".json"):
        with open(savedir + "/evolution_stats_"+t+".json", "r") as f:
            prev_stats[t] = json.loads(f.read())
prev_train_msg = ""
if print_visuals == True:
    app = Ursina()

    window.borderless = False
    window.fullscreen = False
    window.exit_button.visible = False
    window.fps_counter.enabled = False

    s = gs.scaling
    for ypos, row in enumerate(gs.game_space):
        for xpos, item in enumerate(row):
            x, y = gs.grid_to_abs(xpos, ypos)
            cube = Entity(model='cube',
                          color=color.white,
                          scale=(s,s,s),
                          position = (x, y, 0),
                          texture='grass')
    # Make ball
    gs.ball.entity = Entity(model='sphere',
                            color=color.white,
                            scale=(s,s,s),
                            position = (gs.ball.x, gs.ball.y, (-1*gs.scaling)),
                            texture="football_texture")

    for t in gs.agent_types:
        for index, agent in enumerate(gs.agents[t]):
            xabs = gs.agents[t][index].xpos
            yabs = gs.agents[t][index].ypos
            s = gs.scaling * 1
            texture = gs.textures[t]
            gs.agents[t][index].entity = Entity(model='cube',
                                                color=color.white,
                                                scale=(s,s,0),
                                                position = (xabs, yabs, (-1*gs.scaling)),
                                                texture=texture)

    camera.position -= (0, 0, 50*gs.scaling)
    #camera.position = (0, (-45*gs.scaling), (-40*gs.scaling))
    #camera.rotation = (-47, 0, 0)
    app.run()

else:
    while True:
        gs.step()
        if gs.steps % gs.max_episode_len == 0:
            os.system('clear')
            print()
            print(msg(gs))
            print(prev_train_msg)
            print()
            for t in gs.agent_types:
                s0, u0 = gs.get_genome_statistics(t)
                if u0 < 70:
                    prev_train_msg = ""
                    for tt in gs.agent_types:
                        f, m = gs.get_genome_fitness(tt)
                        prev_stats[tt].append(f)
                        with open(savedir + "/evolution_stats_"+tt+".json", "w") as f:
                            f.write(json.dumps(prev_stats[tt]))
                        prev_train_msg += gs.create_new_genome_pool(tt)
                        gs.save_genomes(tt)
                        gs.save_best_policies(tt)
