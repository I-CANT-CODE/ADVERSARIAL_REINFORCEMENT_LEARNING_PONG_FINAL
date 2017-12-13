'''
ONCE THE ABOVE TWO CODES HAVE BEEN RUN FOR 500,000 TIME STEPS EACH, THEN THIS CAN BE RUN
TO EVALUATE AND COMPARE HOW THEY DO.
'''
import pygame
import numpy as np
import math
import random

Results_file = open('Multi_VS_Single_Agent_Results','a')

#MACHINE LEARNING STUFF---------------------------------------------------------
import tensorflow as tf
def NN(x, reuse = False):
    #,
    x = tf.layers.dense(x,units = 512,activation = tf.nn.relu,  name = 'FC1', reuse = reuse)
    x = tf.layers.dense(x,units = 1024,activation = tf.nn.relu, name = 'FC2', reuse = reuse)
    x = tf.layers.dense(x,units = 2048,activation = tf.nn.relu,  name = 'FC3', reuse = reuse)
    x = tf.layers.dense(x,units = 1024,activation = tf.nn.relu, name = 'FC4', reuse = reuse)
    x = tf.layers.dense(x,units = 512,activation = tf.nn.relu, name = 'FC5', reuse = reuse)
    Action_Vals = tf.layers.dense(x,units = 3, name = 'FC6', reuse = reuse)
    return Action_Vals

session = tf.Session()
session.run(tf.global_variables_initializer())
State_In = tf.placeholder(tf.float32, shape = [None, 6])
with tf.variable_scope("paddle"):
    Q = NN(State_In, reuse = False)
saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='paddle'))
saver1.restore(session, 'SINGLE_AGENT_MODEL-500000')

State_In2 = tf.placeholder(tf.float32, shape = [None, 6])
with tf.variable_scope("paddle2"):
    Q2 = NN(State_In2, reuse = False)
saver2 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='paddle2'))
saver2.restore(session, 'MULTIAGENT_MODEL-500000')




GAMMA = .9
EPSILON = 1
training_data = []

#-------------------------------------------------------------------------------


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
PADDLE_SPEED = 6
INIT_BALL_SPEED = 5.50
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
time_step = -1
reward_sum = 0
reward_sum2 = 0
margin = 0
Num_Episodes = 0
Num_Games = 0
while not gameExit:
    time_step = time_step + 1
    REW = 0
    REW2 = 0
    
    S_0 = [BALL_X, BALL_Y, BALL_V_X, BALL_V_Y, PADDLE_LEFT_Y, PADDLE_RIGHT_Y]
    #clock.tick(60)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            #saver.save(session, './most_recent_model')
            
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

    #margin = random.randint(-5,5)
    PADDLE_RIGHT_ACTION = [0,0,0]
    
    
    action_values = session.run(Q,feed_dict = {State_In:[S_0]})
    PADDLE_RIGHT_ACTION[np.argmax(action_values)]=1
        
    PADDLE_LEFT_ACTION = [0,0,0]
    
    action_values = session.run(Q2,feed_dict = {State_In2:[S_0]})
    PADDLE_LEFT_ACTION[np.argmax(action_values)]=1
        
                
    
    #print('here 1?')
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

    #print('here 2?')
    if LEFT_COLLISION:
        REW2 = .1
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
        REW = .1
    if CEILING_COLLISION:
        BALL_Y = 0
        BALL_V_Y = BALL_V_Y * -1
    if FLOOR_COLLISION:
        BALL_Y = WIN_DIM-BALL_DIM
        BALL_V_Y = BALL_V_Y * -1
    if LEFT_PADDLE_FAIL:
        Num_Episodes = Num_Episodes +1
        BALL_SPEED = INIT_BALL_SPEED
        PADDLE_LEFT_Y = PADDLE_RIGHT_Y = WIN_DIM/2-PADDLE_H/2
        BALL_X = WIN_DIM/5
        BALL_Y = WIN_DIM/2
        rand_theta = random.uniform(-.8,.8)
        BALL_V_X = BALL_SPEED*math.cos(rand_theta)
        BALL_V_Y = BALL_SPEED*-math.sin(rand_theta)
        R_POINTS = R_POINTS + 1
        REW = 1
        REW2 = -1
        print(L_POINTS, R_POINTS)
        #print('score - RIGHT = ', R_POINTS, 'LEFT = ',L_POINTS)
    if RIGHT_PADDLE_FAIL:
        Num_Episodes = Num_Episodes + 1
        BALL_SPEED = INIT_BALL_SPEED
        PADDLE_LEFT_Y = PADDLE_RIGHT_Y = WIN_DIM/2-PADDLE_H/2
        BALL_X = WIN_DIM*4/5
        rand_theta = random.uniform(-.8,.8)
        BALL_V_X = BALL_SPEED*-math.cos(rand_theta)
        BALL_V_Y = BALL_SPEED*-math.sin(rand_theta)
        BALL_Y = WIN_DIM/2
        L_POINTS = L_POINTS + 1
        REW = -1
        REW2 = 1
        print(L_POINTS, R_POINTS)
        #print('score - RIGHT = ', R_POINTS, 'LEFT = ',L_POINTS)
    #bound the paddles' movement
    if PADDLE_RIGHT_Y >= WIN_DIM-PADDLE_H+.5*PADDLE_H:
        PADDLE_RIGHT_Y = WIN_DIM-PADDLE_H+.5*PADDLE_H
    if PADDLE_RIGHT_Y <= -.5*PADDLE_H:
        PADDLE_RIGHT_Y = -.5*PADDLE_H
    if PADDLE_LEFT_Y >= WIN_DIM-PADDLE_H/2:
        PADDLE_LEFT_Y = WIN_DIM-PADDLE_H/2
    if PADDLE_LEFT_Y <= -.5*PADDLE_H:
        PADDLE_LEFT_Y = -.5*PADDLE_H
    if (R_POINTS==21)|(L_POINTS==21):
        string =  str(Num_Games)+','+','+ str(L_POINTS) + ',' + str(R_POINTS)+'; \n'
        print(string)
        Results_file.write(string)
        R_POINTS = 0
        L_POINTS = 0

        Num_Games = Num_Games + 1
    if Num_Games == 50:
        gameExit = True
        
    reward_sum = reward_sum + REW
    reward_sum2 = reward_sum2 + REW2
    
    
    
    gameDisplay.fill(black)#fill black background
    pygame.draw.rect(gameDisplay, white, [PADDLE_RIGHT_X,PADDLE_RIGHT_Y,PADDLE_W,PADDLE_H])#draw first paddle
    pygame.draw.rect(gameDisplay, white, [PADDLE_LEFT_X,PADDLE_LEFT_Y,PADDLE_W,PADDLE_H])#draw first paddle
    pygame.draw.rect(gameDisplay, white, [BALL_X,BALL_Y,BALL_DIM,BALL_DIM])#draw ball
    pygame.display.update()
    
    #print(training_data[-1])
Results_file.close()
pygame.quit()      
quit()

