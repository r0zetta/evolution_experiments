import random, sys, time, os, json, re, pickle, operator, math
import numpy as np
from collections import Counter, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from scipy.spatial import distance

class Net(nn.Module):
    def __init__(self,  state_size,  action_size, frames, hidden, layers,
                 lstm1_w, lstm2_w, fc_w):
        super(Net, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.frames = frames
        self.hidden = hidden
        self.layers = layers
        self.lstm = nn.LSTM(self.state_size,
                            self.hidden,
                            self.layers,
                            bidirectional=False,
                            bias=False)

        self.head = nn.Linear(self.hidden, self.action_size, bias=False)
        self.lstm.weight_ih_l0.data = lstm1_w
        self.lstm.weight_hh_l0.data = lstm2_w
        self.head.weight.data = fc_w

    def forward(self, state, hidden):
        x, h = self.lstm(state, hidden)
        x = x[:, -1]
        x = F.softmax(self.head(x), dim=-1)
        return x, h

    def get_action(self, state, hidden):
        with torch.no_grad():
            ret, h = self.forward(state, hidden)
            action = torch.argmax(ret)
            return action, h

class GN_model:
    def __init__(self, state_size, action_size, frames, hidden, layers, lstm1_w, lstm2_w, fc_w):
        self.policy = Net(state_size, action_size, frames, hidden,
                          layers, lstm1_w, lstm2_w, fc_w)
        self.h = (torch.randn(1, 1, hidden),
                  torch.randn(1, 1, hidden))

    def get_action(self, state):
        action, h = self.policy.get_action(state, self.h)
        self.h = h
        return action


class agent:
    def __init__(self, x, y, state_size, action_size, frames, hidden, layers, genome, gi):
        self.xpos = x
        self.ypos = y
        self.fitness = 0
        self.gi = gi
        fs = state_size*frames
        ls = (fs * hidden)
        fcs = action_size * hidden
        genome_size = ls*2 + fcs
        lstm1_w = torch.Tensor(np.reshape(genome[:ls], (fs, state_size)))
        lstm2_w = torch.Tensor(np.reshape(genome[ls:ls*2], (fs, state_size)))
        fc_w = torch.Tensor(np.reshape(genome[ls*2:], (action_size, hidden)))
        self.model = GN_model(state_size, action_size, frames, hidden,
                              layers, lstm1_w, lstm2_w, fc_w)
        self.frames = frames
        self.prev_states = deque()
        self.state = None
        self.episode_steps = 0
        self.last_success = None
        self.holding_food = False
        self.last_action = 0

    def get_action(self):
        ps = []
        for item in self.prev_states:
            ps.append(np.array(item))
        if len(ps) != self.frames:
            print("ERROR!")
            sys.exit(0)
        return self.model.get_action(torch.Tensor(ps))

    def push_state(self, state):
        self.prev_states.append(state)
        if len(self.prev_states) > self.frames:
            self.prev_states.popleft()

class game_space:
    def __init__(self, width, height, frames=4, hidden=32, layers=1,
                 num_walls=0, num_agents=1, num_food=1, num_queens=1,
                 max_episode_len=200, savedir="save"):
        self.savedir = savedir
        self.max_episode_len = max_episode_len
        self.num_agents = num_agents
        self.width = width+2
        self.height = height+2
        self.num_food = num_food
        self.num_queens = num_queens
        self.berries = []
        self.food = []
        self.queens = []
        self.start_walls = num_walls
        self.action_size = 6
        self.frames = frames
        self.hidden = hidden
        self.layers = layers
        self.pool_size = 2000
        self.state_size = self.get_state_size()
        fs = self.state_size*self.frames
        ls = (fs * self.hidden)
        fcs = self.action_size * self.hidden
        self.genome_size = ls*2 + fcs
        self.agent_types = ["picker", "feeder"]
        self.starts = {"picker":2, "feeder":2000}
        self.food_id = 10000
        self.queen_id = 20000
        self.berry_id = 30000
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
        self.agents = {}
        for t in self.agent_types:
            self.agents[t] = []
            for index in range(self.num_agents):
                self.agents[t].append(None)
        self.walls = []
        space = self.make_empty_game_space()
        self.game_space = space
        self.add_walls(self.start_walls)
        self.num_walls = len(self.walls)
        self.initial_game_space = np.array(self.game_space)
        self.make_food()
        self.make_queens()
        self.make_berries()
        for t in self.agent_types:
            for index, agent in enumerate(self.agents[t]):
                self.create_new_agent(index, t)
        for t in self.agent_types:
            for index, agent in enumerate(self.agents[t]):
                state = self.get_agent_state(index, t)
                for _ in range(self.frames):
                    self.agents[t][index].push_state(state)

    def step(self):
        for t in self.agent_types:
            for index in range(len(self.agents[t])):
                action = gs.agents[t][index].get_action()
                self.move_agent(index, action, t)

    def make_empty_game_space(self):
        space = np.zeros((self.height, self.width), dtype=int)
        for n in range(self.width):
            space[0][n] = 1
            space[1][n] = 1
            space[self.height-1][n] = 1
            space[self.height-2][n] = 1
        for n in range(self.height):
            space[n][0] = 1
            space[n][1] = 1
            space[n][self.width-1] = 1
            space[n][self.width-2] = 1
        return space

    def make_food(self):
        self.food = []
        locations = self.get_random_empty_space(self.num_food)
        for item in locations:
            y, x = item
            self.food.append([y, x])

    def make_queens(self):
        self.queens = []
        locations = self.get_random_empty_space(self.num_queens)
        for item in locations:
            y, x = item
            self.queens.append([y, x])

    def make_berries(self):
        self.berries = []
        locations = self.get_random_empty_space(self.num_queens)
        for item in locations:
            y, x = item
            self.berries.append([y, x])

    def add_blocks(self, num):
        locations = self.get_random_empty_space(num)
        for item in locations:
            y, x = item
            self.game_space[y][x] = 1
            self.walls.append([y, x])
        self.num_walls = len(self.walls)

    def add_walls(self, num):
        added = 0
        while added < num:
            item = self.get_random_empty_space(1)
            ypos, xpos = item[0]
            self.game_space[ypos][xpos] = 1
            self.walls.append([ypos, xpos])
            for n in range(50):
                move = random.randint(0,3)
                if move == 0:
                    xpos = max(0, xpos-1)
                elif move == 1:
                    xpos = min(self.width-1, xpos+1)
                elif move == 2:
                    ypos = max(0, ypos-1)
                elif move == 3:
                    ypos = min(self.height-1, ypos+1)
                if self.game_space[ypos][xpos] == 0:
                    added += 1
                self.game_space[ypos][xpos] = 1
                self.walls.append([ypos, xpos])
                if added >= num:
                    break

    def create_new_agent(self, index, atype):
        item = self.get_random_empty_space(1)
        ypos, xpos = item[0]
        state_size = self.get_state_size()

        selectable = []
        for i, item in enumerate(self.genome_pool[atype]):
            g, f = item
            if f is None:
                selectable.append(i)
        gi = random.choice(selectable)
        item = self.genome_pool[atype][gi]
        genome, fitness = item

        self.agents[atype][index] = agent(xpos, ypos, state_size, self.action_size,
                                          self.frames, self.hidden, self.layers,
                                          genome, gi)

    def add_items_to_game_space(self):
        space = np.array(self.game_space)
        for t in self.agent_types:
            if len(self.agents[t]) > 0:
                for index, agent in enumerate(self.agents[t]):
                    if agent is not None:
                        space[agent.ypos][agent.xpos] = self.starts[t]+index
        for item in self.queens:
            y, x = item
            space[y][x] = self.queen_id
        for item in self.food:
            y, x = item
            space[y][x] = self.food_id
        for item in self.berries:
            y, x = item
            space[y][x] = self.berry_id
        return space

    def get_random_empty_space(self, num=1):
        space = self.add_items_to_game_space()
        empties = list(np.argwhere(space == 0))
        return random.sample(empties, num)

    def respawn_agent(self, index, atype):
        gi = self.agents[atype][index].gi
        item = self.genome_pool[atype][gi]
        genome, fitness = item
        fitness = self.agents[atype][index].fitness
        self.genome_pool[atype][gi] = [genome, fitness]
        self.create_new_agent(index, atype)
        state = self.get_agent_state(index, atype)
        for _ in range(self.frames):
            self.agents[atype][index].push_state(state)

    def spawn_more_berries(self):
        num_berries = len(self.berries)
        if num_berries < 10:
            new_berries = 10-num_berries
            positions = self.get_random_empty_space(new_berries)
            for p in positions:
                y, x = p
                self.add_berry(x, y)

    def add_berry(self, xpos, ypos):
        self.berries.append([ypos, xpos])

    def remove_berry(self, xpos, ypos):
        bi = None
        for index, item in enumerate(self.berries):
            y, x = item
            if y == ypos and x == xpos:
                bi = index
                break
        del self.berries[bi]

    def get_space_val_in_direction(self, xpos, ypos, action):
        newx = xpos
        newy = ypos
        if action == 1: # moving up
            newy = ypos-1
        elif action == 2: # moving right
            newx = xpos+1
        elif action == 3: # moving down
            newy = ypos+1
        elif action == 4: # moving left
            newx = xpos-1
        space = self.add_items_to_game_space()
        space_val = space[newy][newx]
        return space_val

    def move_forward(self, index, action, atype):
        xpos = self.agents[atype][index].xpos
        ypos = self.agents[atype][index].ypos
        holding = self.agents[atype][index].holding_food
        got_food = False
        newx = xpos
        newy = ypos
        if action == 1: # moving up
            newy = ypos-1
        elif action == 2: # moving right
            newx = xpos+1
        elif action == 3: # moving down
            newy = ypos+1
        elif action == 4: # moving left
            newx = xpos-1
        space = self.add_items_to_game_space()
        space_val = space[newy][newx]
        if space_val == 0: # empty space, move forward
            self.agents[atype][index].xpos = newx
            self.agents[atype][index].ypos = newy
        if atype == "feeder":
            if space_val == self.berry_id and holding == False:
                self.agents[atype][index].xpos = newx
                self.agents[atype][index].ypos = newy
                self.agents[atype][index].holding_food = True
                self.remove_berry(newx, newy)
                got_food = True
        return newx, newy, space_val, got_food

    def update_success_step(self, index, atype):
        s = self.agents[atype][index].episode_steps
        self.agents[atype][index].last_success = s

    def move_agent(self, index, action, atype):
        done = False
        holding = self.agents[atype][index].holding_food
        if action == 0: # do nothing
            pass
        elif action in [1, 2, 3, 4]:
            newx, newy, space_val, got_food = self.move_forward(index, action, atype)
            if space_val == self.food_id:
                if atype == "picker" and holding == False:
                    self.agents[atype][index].holding_food = True
            if space_val == self.queen_id:
                if atype == "feeder" and holding == True:
                    self.agents[atype][index].fitness += 10
                    self.update_success_step(index, atype)
                    self.agents[atype][index].holding_food = False
        else: # drop berry
            if atype == "picker":
                if holding == True:
                    xpos = self.agents[atype][index].xpos
                    ypos = self.agents[atype][index].ypos
                    last_action = self.agents[atype][index].last_action
                    if last_action < 1:
                        last_action = 1
                    sv = self.get_space_val_in_direction(xpos, ypos, last_action)
                    if sv == 0:
                        self.agents[atype][index].holding_food = False
                        self.add_berry(xpos, ypos)
                        self.agents[atype][index].fitness += 10
                        self.update_success_step(index, atype)

        if action in [1, 2, 3, 4]:
            self.agents[atype][index].last_action = action
        self.agents[atype][index].fitness += 1
        state = self.get_agent_state(index, atype)
        self.agents[atype][index].push_state(state)
        self.agents[atype][index].episode_steps += 1
        ls = self.agents[atype][index].last_success
        if ls is not None:
            s = self.agents[atype][index].episode_steps
            if (s - ls) > self.max_episode_len:
                self.respawn_agent(index, atype)
        elif self.agents[atype][index].episode_steps >= self.max_episode_len:
            self.respawn_agent(index, atype)


    def get_agent_at_position(self, xpos, ypos, atype):
        for index, agent in enumerate(self.agents[atype]):
            if agent.xpos == xpos and agent.ypos == ypos:
                return index
        return None

    def get_agent_state(self, index, atype):
        return self.make_small_state(index, atype)

    def get_tile_val(self, tile, atype):
        if tile == 0:
            return 0
        if tile == 1:
            return 1
        if tile >= self.starts["picker"] and tile < self.starts["feeder"]:
            return 2
        if tile >= self.starts["feeder"] and tile < self.food_id:
            return 3
        if tile == self.food_id:
            return 4
        if tile == self.queen_id:
            return 5
        if tile == self.berry_id:
            return 6

    def distance(self, xa, ya, xb, yb):
        dst = distance.euclidean([xa, ya], [xb, yb])
        return dst

    def get_nearest_obj_offset(self, xpos, ypos, itemid):
        sp = self.add_items_to_game_space()
        obj_positions = np.argwhere(sp==itemid)
        if len(obj_positions) < 1:
            return 0,0
        ind = {}
        for index, item in enumerate(obj_positions):
            ey, ex = item
            dist = self.distance(ex, ey, xpos, ypos)
            ind[index] = dist
        nearest_index = None
        for index, dist in sorted(ind.items(), key=operator.itemgetter(1),reverse=False):
            nearest_index = index
            break
        item = obj_positions[nearest_index]
        ey, ex = item
        xoff = ex - xpos
        yoff = ey - ypos
        return xoff, yoff

    def get_state_size(self):
        state_size = 32
        return state_size

    def make_small_state(self, index, atype):
        xpos = self.agents[atype][index].xpos
        ypos = self.agents[atype][index].ypos
        holding = int(self.agents[atype][index].holding_food)
        la = self.agents[atype][index].last_action
        fx, fy = self.get_nearest_obj_offset(xpos, ypos, self.food_id)
        bx, by = self.get_nearest_obj_offset(xpos, ypos, self.berry_id)
        qx, qy = self.get_nearest_obj_offset(xpos, ypos, self.queen_id)

        space = self.add_items_to_game_space()
        os = [-2, -1, 0, 1, 2]
        offsets = []
        for x in os:
            for y in os:
                if x == 0 and y == 0:
                    continue
                offsets.append([x, y])

        tiles = [la, holding, fx, fy, bx, by, qx, qy]
        for i in offsets:
            oy, ox = i
            tile = self.get_tile_val(space[ypos+oy][xpos+ox], atype)
            tiles.append(tile)
        state = tiles
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
            n = int(0.001 * len(g))
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
            return self.max_episode_len
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
        if fitness < self.max_episode_len:
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
                if fitness > self.max_episode_len:
                    vi[index] = fitness
        count = 0
        for index, fitness in sorted(vi.items(), key=operator.itemgetter(1),reverse=True):
            genome = pool[index][0]
            fit_genomes.append(genome)
            self.add_best_fitness_entry(atype, genome, fitness)
            count += 1
            if len(vi) > 50:
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
        threshold = 0.20
        fit_genomes = self.get_best_genomes(atype, threshold)
        best_policies = []
        if len(self.best_policies[atype]) > 50:
            best_policies = self.get_best_policies(atype, 50)
        msg += atype + ": Previous pool had " + str(len(fit_genomes)) + " fit genomes.\n"
        mutated_fit = []
        for item in fit_genomes:
            if len(fit_genomes) > 10:
                mutated_fit.extend(self.mutate_genome(item, 3))
            else:
                mutated_fit.extend(self.mutate_genome(item, 10))
        msg += "New genomes from mutations: " + str(len(mutated_fit)) + "\n"
        # Select pairs to reproduce and mutate
        repr_genomes = []
        if len(fit_genomes) > 1:
            num_pairs = min(int(self.pool_size/50), int(len(fit_genomes)))
            for _ in range(num_pairs):
                g1, g2 = random.sample(fit_genomes, 2)
                offspring = (self.reproduce_genome(g1, g2, 4))
                repr_genomes.extend(offspring)
                for item in offspring:
                    repr_genomes.extend(self.mutate_genome(item, 5))
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
                if fitness > self.max_episode_len:
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




    def get_printable(self, item):
        if item == 1:
            return "\x1b[1;37;40m" + "░" + "\x1b[0m"
        elif item >= self.starts["picker"] and item < self.starts["feeder"]:
            return "\x1b[1;32;40m" + "x" + "\x1b[0m"
        elif item >= self.starts["feeder"] and item < self.food_id:
            return "\x1b[1;33;40m" + "x" + "\x1b[0m"
        elif item == self.food_id:
            return "\x1b[1;31;40m" + "¥" + "\x1b[0m"
        elif item == self.queen_id:
            return "\x1b[1;36;40m" + "O" + "\x1b[0m"
        elif item == self.berry_id:
            return "\x1b[1;35;40m" + "o" + "\x1b[0m"
        else:
            return "\x1b[1;32;40m" + " " + "\x1b[0m"

    def print_game_space(self):
        printable = ""
        space = self.add_items_to_game_space()
        for column in space:
            for item in column:
                printable += self.get_printable(item)
            printable += "\n"
        return printable


# Train the game
def msg(gs):
    msg = "Steps: " + str(steps) + " Episode length: " + str(gs.max_episode_len) 
    msg += " Genome size: " + str(gs.genome_size) 
    msg += " Pool size: " + str(gs.pool_size) + "\n\n"
    for t in gs.agent_types:
        s, u = gs.get_genome_statistics(t)
        f, m = gs.get_genome_fitness(t)
        pf, pn = gs.get_best_policy_stats(t)
        msg += t + ": Success: " + str(s) + " Unused: " + str(u) 
        msg += " Fitness: " + "%.2f"%f + " Max: " + str(m) + "\n"
        msg += "Best policy fitness: " + "%.2f"%pf + " Num: " + str(pn) + "\n"
        msg += "[ "
        p = prev_stats[t]
        for s in p[-10:]:
            msg += "%.2f"%s + " "
        msg += "]\n\n"
    return msg

random.seed(1337)

game_space_width = 30
game_space_height = 15
num_walls = 20
num_agents = 10
num_food = 10
num_queens = 10
max_episode_len = 50
hidden = 32
frames = 4
layers=1
savedir = "coop_feed_rnn_save"
if not os.path.exists(savedir):
    os.makedirs(savedir)

gs = game_space(game_space_width,
                game_space_height,
                frames=frames,
                hidden=hidden,
                layers=layers,
                num_walls=num_walls,
                num_agents=num_agents,
                num_food=num_food,
                num_queens=num_queens,
                max_episode_len=max_episode_len,
                savedir=savedir)

print_visuals = False
steps = 0
prev_stats = {}
for t in gs.agent_types:
    prev_stats[t] = []
    if os.path.exists(savedir + "/evolution_stats_"+t+".json"):
        with open(savedir + "/evolution_stats_"+t+".json", "r") as f:
            prev_stats[t] = json.loads(f.read())
prev_train_msg = ""
while True:
    gs.step()
    if print_visuals == True:
        os.system('clear')
        print()
        print(gs.print_game_space())
        print()
        print(msg(gs))
        time.sleep(0.03)
    elif steps%gs.max_episode_len == 0:
        os.system('clear')
        print()
        print(msg(gs))
        print(prev_train_msg)
        print()
    if steps % int(gs.max_episode_len) == 0:
        gs.spawn_more_berries()
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
    steps += 1

