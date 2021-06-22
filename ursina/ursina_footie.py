from ursina import *
import numpy as np
import random

scaling = 6

UP = 0
UP_RIGHT = 1
RIGHT = 2
DOWN_RIGHT = 3
DOWN = 4
DOWN_LEFT = 5
LEFT = 6
UP_LEFT = 7
DIRECTIONS = [UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, LEFT, UP_LEFT]

class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = 0
        self.speed = 0
        self.friction = 0.95
        self.hit_speed = 2
        self.hit_cooldown = 0
        s = scaling
        self.ball = Entity(model='sphere',
                           color=color.white,
                           scale=(s, s, s),
                           position = (self.x, self.y, (-1*scaling)),
                           texture="football_texture")



def move_item(x, y, direction, speed, top, bottom, left, right):
    nx = x
    ny = y
    if direction == UP:
        ny = ny-speed
    elif direction == UP_RIGHT:
        ny = ny-speed
        nx = nx+speed
    elif direction == RIGHT:
        nx = nx+speed
    elif direction == DOWN_RIGHT:
        nx = nx+speed
        ny = ny+speed
    elif direction == DOWN:
        ny = ny+speed
    elif direction == DOWN_LEFT:
        ny = ny+speed
        nx = nx-speed
    elif direction == LEFT:
        nx = nx-speed
    elif direction == UP_LEFT:
        ny = ny-speed
        nx = nx-speed
    if nx < left:
        nx = left
    if nx > right:
        nx = right
    if ny > bottom:
        ny = bottom
    if ny < top:
        ny = top
    return nx, ny

def abs_to_grid(absx, absy):
    nx = int(round(absx/scaling)) + int(round(width/2))
    ny = int(round(absy/scaling)) + int(round(height/2))
    return nx, ny

def grid_to_abs(gridx, gridy):
    x = scaling * (gridx-int(round(width/2)))
    y = scaling * (gridy-int(round(height/2)))
    return x, y

def move_dudes():
    top = 0
    right = width-2
    bottom = height-2
    left = 0
    for index, dude in enumerate(dudes):
        cx, cy = dude_abs_positions[index]
        dx, dy = dude_new_abs_positions[index]
        if cx == dx and cy == dy:
            if random.random() > 0.8:
                # Move to ball
                dude_new_abs_positions[index] = [int(b.x), int(b.y)]
            else:
                # Choose a new random location to move to
                tx, ty = abs_to_grid(cx, cy)
                direction = random.choice(DIRECTIONS)
                speed = random.choice([1,2,3,4,5,6,7,8])
                nx, ny = move_item(tx, ty, direction, speed, top, bottom, left, right)
                dude_directions[index] = direction
                x, y = grid_to_abs(nx, ny)
                dude_new_abs_positions[index]=[x, y]
        else:
            if cx < dx:
                cx += 1
            elif cx > dx:
                cx -= 1
            if cy < dy:
                cy += 1
            elif cy > dy:
                cy -= 1
        dude.position = (cx, cy, (-1*scaling))
        dude_abs_positions[index] = [cx, cy]

def ball_collision():
    bottom = int((height*scaling)/2) - scaling
    top = -1 * bottom
    right = int((width*scaling)/2) - scaling
    left = -1 * right
    hit_directions = []
    who_hit = []
    # Check for collision with sides of pitch
    if b.y <= top:
        if b.direction == UP:
            hit_directions.append(DOWN)
        elif b.direction == UP_RIGHT:
            hit_directions.append(DOWN_RIGHT)
        elif b.direction == UP_LEFT:
            hit_directions.append(DOWN_LEFT)
    elif b.y >= bottom:
        if b.direction == DOWN:
            hit_directions.append(UP)
        elif b.direction == DOWN_RIGHT:
            hit_directions.append(UP_RIGHT)
        elif b.direction == DOWN_LEFT:
            hit_directions.append(UP_LEFT)
    if b.x <= left:
        if b.direction == LEFT:
            hit_directions.append(RIGHT)
        elif b.direction == UP_LEFT:
            hit_directions.append(UP_RIGHT)
        elif b.direction == DOWN_LEFT:
            hit_directions.append(DOWN_RIGHT)
    elif b.x >= right:
        if b.direction == RIGHT:
            hit_directions.append(LEFT)
        elif b.direction == DOWN_RIGHT:
            hit_directions.append(DOWN_LEFT)
        elif b.direction == UP_RIGHT:
            hit_directions.append(UP_LEFT)
    # Check for collisions with players
    lt = -1 * scaling
    lt1 = -1 * int(round(scaling/2))
    ut = scaling
    ut1 = int(round(scaling/2))
    for index, dude in enumerate(dudes):
        dx, dy = dude_abs_positions[index]
        ox = dx - b.x
        oy = dy - b.y
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
            who_hit.append(index)
    return hit_directions, who_hit

def move_ball():
    # See if it collided with anything
    if b.hit_cooldown == 0:
        hit_directions, who_hit = ball_collision()
        if len(hit_directions) > 0:
            b.direction = random.choice(hit_directions)
            if len(who_hit) > 0:
                b.speed = b.hit_speed
            b.hit_cooldown = 5
    # Move ball
    bottom = int((height*scaling)/2) - scaling
    top = -1 * bottom
    right = int((width*scaling)/2) - scaling
    left = -1 * right
    b.x, b.y = move_item(b.x, b.y, b.direction, b.speed, top, bottom, left, right)
    b.speed = b.speed * b.friction
    b.ball.position = (b.x, b.y, (-1*scaling))
    b.hit_cooldown -= 1
    if b.hit_cooldown < 0:
        b.hit_cooldown = 0

def update():
    move_dudes()
    move_ball()
    direction = b.direction
    speed = b.speed
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
    b.ball.rotation = (rotx, roty, 0)
    if held_keys['q']:
        camera.position += (0, 0, time.dt*10)
    if held_keys['e']:
        camera.position -= (0, 0, time.dt*10)
    if held_keys['a']:
        camera.position += (time.dt*50, 0, 0)
    if held_keys['d']:
        camera.position -= (time.dt*50, 0, 0)
    if held_keys['w']:
        camera.position -= (0, time.dt*50, 0)
    if held_keys['s']:
        camera.position += (0, time.dt*50, 0)
    if held_keys['i']:
        camera.rotation_z += 1
    if held_keys['k']:
        camera.rotation_z -= 1
    if held_keys['j']:
        camera.rotation_x += 1
    if held_keys['l']:
        camera.rotation_x -= 1
    if held_keys['u']:
        camera.rotation_y += 1
    if held_keys['o']:
        camera.rotation_y -= 1
    #print(camera.position, camera.rotation)

def get_random_empty_space(space, num=1):
    empties = list(np.argwhere(space == 0))
    return random.sample(empties, num)

frames = 0
app = Ursina()

window.title = 'Whatever'
window.borderless = False
window.fullscreen = False
window.exit_button.visible = False
window.fps_counter.enabled = True

space = np.zeros((15,25))
height, width = space.shape

# Ball
b = Ball(0, 0)

# Playing field
count = 1
for ypos, row in enumerate(space):
    for xpos, item in enumerate(row):
        x, y = grid_to_abs(xpos, ypos)
        s = scaling
        if item == 1:
            cube = Entity(model='cube',
                          color=color.white,
                          scale=(s,s,s),
                          position = (x, y, (-1*scaling)),
                          texture='rock')
        cube = Entity(model='cube',
                      color=color.white,
                      scale=(s,s,s),
                      position = (x, y, 0),
                      texture="grass")

# Players
num_dudes = 10
dudes = []
dude_abs_positions = []
dude_new_abs_positions = []
dude_directions = []
for n in range(num_dudes):
    texture = "yak"
    xpos = random.choice(range(0, int(width/2)))
    if n >= int(num_dudes/2):
        xpos = random.choice(range(int(width/2), width))
        texture = "hog"
    ypos = random.choice(range(height))
    x, y = grid_to_abs(xpos, ypos)
    dude_abs_positions.append([x, y])
    dude_new_abs_positions.append([x, y])
    dude_directions.append(0)
    s = scaling
    cube = Entity(model='cube',
                  color=color.white,
                  scale=(s,s,0),
                  position = (x, y, (-1*scaling)),
                  texture=texture)
    dudes.append(cube)


camera.position = (0, 0, -60*scaling)
#camera.position = (0, (-45*scaling), (-40*scaling))
#camera.rotation = (-47, 0, 0)
app.run()
