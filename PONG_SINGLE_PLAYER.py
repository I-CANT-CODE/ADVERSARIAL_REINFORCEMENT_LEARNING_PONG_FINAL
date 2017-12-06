import pygame
import numpy as np
import math
import random

#CONSTANTS
WIN_DIM = 320
PADDLE_W = 10
PADDLE_H = 70
BALL_DIM = 10

#PADDLE LEFT
PADDLE_LEFT_X = PADDLE_W
PADDLE_LEFT_Y_INIT = WIN_DIM/2
PADDLE_LEFT_Y = PADDLE_LEFT_Y_INIT

#PADDLE RIGHT
PADDLE_RIGHT_X = WIN_DIM-2*PADDLE_W
PADDLE_RIGHT_Y_INIT = WIN_DIM/2
PADDLE_RIGHT_Y = PADDLE_RIGHT_Y_INIT

#BALL VARIABLES
BALL_X_INIT = BALL_Y_INIT = WIN_DIM/2
BALL_X = BALL_X_INIT
BALL_Y = BALL_Y_INIT

#BALL VELOCITIES
BALL_V_X = 2
BALL_V_Y = 2

#SPEEDS
PADDLE_SPEED = 5
INIT_BALL_SPEED = 4
BALL_SPEED = INIT_BALL_SPEED
COLLISION_MARGIN = 10

#PADDLE ACTIONS
UP = [1,0,0]
DONT_MOVE = [0,1,0]
DOWN = [0,0,1]

#COLORS
white = (255,255,255)
black = (0,0,0)

#initialize game loop
gameDisplay = pygame.display.set_mode([WIN_DIM,WIN_DIM])
gameExit = False
PADDLE_LEFT_ACTION=PADDLE_RIGHT_ACTION=DONT_MOVE
clock = pygame.time.Clock()
L_POINTS = 0
R_POINTS = 0
margin = random.randint(0,5)
while not gameExit:
    clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            gameExit = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_DOWN:
                PADDLE_RIGHT_ACTION = DOWN
            elif event.key == pygame.K_UP:
                PADDLE_RIGHT_ACTION = UP
            elif event.key == pygame.K_w:
                PADDLE_LEFT_ACTION = UP
            elif event.key == pygame.K_s:
                PADDLE_LEFT_ACTION = DOWN
        if event.type == pygame.KEYUP:
            if (event.key ==pygame.K_DOWN)|(event.key ==pygame.K_UP):
                PADDLE_RIGHT_ACTION = DONT_MOVE
            if (event.key ==pygame.K_s)|(event.key ==pygame.K_w):
                PADDLE_LEFT_ACTION = DONT_MOVE

    
    if (BALL_V_X<0)&(BALL_X<WIN_DIM*.50):
        if (PADDLE_LEFT_Y+PADDLE_H/2)>BALL_Y+.5*BALL_DIM+margin:
            PADDLE_LEFT_ACTION = UP
        elif (PADDLE_LEFT_Y+PADDLE_H/2)<BALL_Y+.5*BALL_DIM-margin:
            PADDLE_LEFT_ACTION = DOWN
        else:
            PADDLE_LEFT_ACTION = DONT_MOVE
    else:
        PADDLE_LEFT_ACTION = DONT_MOVE
                
    
    
    #ACTION CHECK
    if np.argmax(PADDLE_RIGHT_ACTION)==np.argmax(UP):
        PADDLE_RIGHT_Y = PADDLE_RIGHT_Y-PADDLE_SPEED
    elif np.argmax(PADDLE_RIGHT_ACTION)==np.argmax(DOWN):
        PADDLE_RIGHT_Y = PADDLE_RIGHT_Y+PADDLE_SPEED
    elif np.argmax(PADDLE_RIGHT_ACTION)==np.argmax(DONT_MOVE):
        PADDLE_RIGHT_Y = PADDLE_RIGHT_Y
    if np.argmax(PADDLE_LEFT_ACTION)==np.argmax(UP):
        PADDLE_LEFT_Y = PADDLE_LEFT_Y-PADDLE_SPEED
    elif np.argmax(PADDLE_LEFT_ACTION)==np.argmax(DOWN):
        PADDLE_LEFT_Y = PADDLE_LEFT_Y+PADDLE_SPEED
    elif np.argmax(PADDLE_LEFT_ACTION)==np.argmax(DONT_MOVE):
        PADDLE_LEFT_Y = PADDLE_LEFT_Y
    BALL_X = BALL_X + BALL_V_X
    BALL_Y = BALL_Y + BALL_V_Y
    #DEFINE COLLISION CASES:
    LEFT_COLLISION = (BALL_X<(PADDLE_LEFT_X+PADDLE_W))&(BALL_X>PADDLE_LEFT_X)&((BALL_Y+BALL_DIM)>PADDLE_LEFT_Y)&(BALL_Y<(PADDLE_LEFT_Y+PADDLE_H))
    RIGHT_COLLISION = (BALL_X>(PADDLE_RIGHT_X-BALL_DIM))&(BALL_X<(PADDLE_RIGHT_X+PADDLE_W))&((BALL_Y+BALL_DIM)>PADDLE_RIGHT_Y)&(BALL_Y<(PADDLE_RIGHT_Y+PADDLE_H))
    LEFT_PADDLE_FAIL = BALL_X+BALL_DIM<=0
    RIGHT_PADDLE_FAIL = BALL_X> WIN_DIM
    FLOOR_COLLISION = BALL_Y>(WIN_DIM-BALL_DIM)
    CEILING_COLLISION = BALL_Y<0
    
    if LEFT_COLLISION:
        margin = random.randint(0,35) 
        BALL_SPEED = BALL_SPEED + .1
        BALL_X = PADDLE_LEFT_X+PADDLE_W
        
        BALL_PADDLE_LEFT_COORDINATE = BALL_Y + BALL_DIM/2 - PADDLE_LEFT_Y
        if BALL_PADDLE_LEFT_COORDINATE < 0:
            BALL_PADDLE_LEFT_COORDINATE = 0
        if BALL_PADDLE_LEFT_COORDINATE > PADDLE_H:
            BALL_PADDLE_LEFT_COORDINATE = PADDLE_H
        #convert from [0,70] to [1.309,-1.309]
        G = BALL_PADDLE_LEFT_COORDINATE/70
        BALL_PADDLE_LEFT_COORDINATE = .8*(1-G)-.8*(G)
            

        
        BALL_V_X = BALL_SPEED*math.cos(BALL_PADDLE_LEFT_COORDINATE)
        BALL_V_Y = BALL_SPEED*-math.sin(BALL_PADDLE_LEFT_COORDINATE)
    if RIGHT_COLLISION:
        BALL_SPEED = BALL_SPEED + .1
        BALL_X = PADDLE_RIGHT_X-BALL_DIM
        
        BALL_PADDLE_RIGHT_COORDINATE = BALL_Y + BALL_DIM/2 - PADDLE_RIGHT_Y
        if BALL_PADDLE_RIGHT_COORDINATE < 0:
            BALL_PADDLE_RIGHT_COORDINATE = 0
        if BALL_PADDLE_RIGHT_COORDINATE > PADDLE_H:
            BALL_PADDLE_RIGHT_COORDINATE = PADDLE_H
        #convert from [0,70] to [1.8326,4.45059]
        G = BALL_PADDLE_RIGHT_COORDINATE/70
        BALL_PADDLE_RIGHT_COORDINATE = .8*(1-G)-.8*(G)

        
        BALL_V_X = BALL_SPEED*-math.cos(BALL_PADDLE_RIGHT_COORDINATE)
        BALL_V_Y = BALL_SPEED*-math.sin(BALL_PADDLE_RIGHT_COORDINATE)
    if CEILING_COLLISION:
        
        BALL_Y = 0
        BALL_V_Y = BALL_V_Y * -1
    if FLOOR_COLLISION:
        BALL_Y = WIN_DIM-BALL_DIM
        BALL_V_Y = BALL_V_Y * -1
    if LEFT_PADDLE_FAIL:
        BALL_SPEED = INIT_BALL_SPEED
        PADDLE_LEFT_Y = PADDLE_RIGHT_Y = WIN_DIM/2
        BALL_X = WIN_DIM/5
        BALL_Y = WIN_DIM/2
        rand_theta = random.uniform(-.8,.8)
        BALL_V_X = BALL_SPEED*math.cos(rand_theta)
        BALL_V_Y = BALL_SPEED*-math.sin(rand_theta)
        R_POINTS = R_POINTS + 1
        print('score - RIGHT = ', R_POINTS, 'LEFT = ',L_POINTS)
    if RIGHT_PADDLE_FAIL:
        BALL_SPEED = INIT_BALL_SPEED
        PADDLE_LEFT_Y = PADDLE_RIGHT_Y = WIN_DIM/2
        BALL_X = WIN_DIM*4/5
        rand_theta = random.uniform(-.8,.8)
        BALL_V_X = BALL_SPEED*-math.cos(rand_theta)
        BALL_V_Y = BALL_SPEED*-math.sin(rand_theta)
        BALL_Y = WIN_DIM/2
        L_POINTS = L_POINTS + 1
        print('score - RIGHT = ', R_POINTS, 'LEFT = ',L_POINTS)
    #bound the paddles' movement
    if PADDLE_RIGHT_Y >= WIN_DIM-PADDLE_H+.5*PADDLE_H:
        PADDLE_RIGHT_Y = WIN_DIM-PADDLE_H+.5*PADDLE_H
    if PADDLE_RIGHT_Y <= -.5*PADDLE_H:
        PADDLE_RIGHT_Y = -.5*PADDLE_H
    if PADDLE_LEFT_Y >= WIN_DIM-PADDLE_H/2:
        PADDLE_LEFT_Y = WIN_DIM-PADDLE_H/2
    if PADDLE_LEFT_Y <= -.5*PADDLE_H:
        PADDLE_LEFT_Y = -.5*PADDLE_H

    
    
    
    gameDisplay.fill(black)#fill black background
    pygame.draw.rect(gameDisplay, white, [PADDLE_RIGHT_X,PADDLE_RIGHT_Y,PADDLE_W,PADDLE_H])#draw first paddle
    pygame.draw.rect(gameDisplay, white, [PADDLE_LEFT_X,PADDLE_LEFT_Y,PADDLE_W,PADDLE_H])#draw first paddle
    pygame.draw.rect(gameDisplay, white, [BALL_X,BALL_Y,BALL_DIM,BALL_DIM])#draw ball
    pygame.display.update()

pygame.quit()
quit()
    

    
    

    
                
            

                
                
                

