import pygame
import tensorflow as tf
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to pre-trained caffemodel file")
ap.add_argument("-th", "--threshold", type=float, default=0.20,
                help="probability threshold to ignore false detections")
args = vars(ap.parse_args())

#model = tf.keras.models.load_model('model_new_4pm.h5')
model = tf.keras.models.load_model('model_gd_99.h5')
video = cv2.VideoCapture(0)

from keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet import preprocess_input

pygame.init()
point_selected = False
point = ()
old_points = np.array([[]])

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


class player:
        """docstring for ClassName"""
        def __init__(self, player_number):

                self.walkcount = 0
                self.punch_kick_count = 0
                self.walk_left = False
                self.walk_right = False
                self.kick = False
                self.punch = False
                self.player_width = 150
                self.pos_x= [ ]
                self.character = pygame.image.load( str(player_number) + '/character/001.png')
                self.walk_left_images  =  [ pygame.image.load(str(player_number) + '/walk/001.png'), pygame.image.load(str(player_number) + '/walk/002.png'),pygame.image.load(str(player_number) + '/walk/003.png'),pygame.image.load(str(player_number) + '/walk/004.png'),pygame.image.load(str(player_number) + '/walk/005.png'),pygame.image.load(str(player_number) + '/walk/006.png'),pygame.image.load(str(player_number) + '/walk/007.png'),pygame.image.load(str(player_number) + '/walk/008.png'),pygame.image.load(str(player_number) + '/walk/009.png') ]
                self.walk_right_images =   [ pygame.image.load(str(player_number) + '/walk/009.png'), pygame.image.load(str(player_number) + '/walk/008.png'),pygame.image.load(str(player_number) + '/walk/007.png'),pygame.image.load(str(player_number) + '/walk/006.png'),pygame.image.load(str(player_number) + '/walk/005.png'),pygame.image.load(str(player_number) + '/walk/004.png'),pygame.image.load(str(player_number) + '/walk/003.png'),pygame.image.load(str(player_number) + '/walk/002.png'),pygame.image.load(str(player_number) + '/walk/001.png') ]
                self.punch_images = [pygame.image.load(str(player_number) + '/punch/1.png'), pygame.image.load(str(player_number) + '/punch/2.png'),pygame.image.load(str(player_number) + '/punch/3.png'),pygame.image.load(str(player_number) + '/punch/4.png')]
                self.kick_images = [pygame.image.load(str(player_number) + '/kick/1.png'), pygame.image.load(str(player_number) + '/kick/2.png'),pygame.image.load(str(player_number) + '/kick/3.png'),pygame.image.load(str(player_number) + '/kick/4.png')]
                self.hurt_images = [pygame.image.load(str(player_number) + '/hurt/1.png'), pygame.image.load(str(player_number) + '/hurt/2.png'),pygame.image.load(str(player_number) + '/hurt/3.png'),pygame.image.load(str(player_number) + '/hurt/4.png'), pygame.image.load(str(player_number) + '/hurt/5.png')]
                self.pos_x= 280 if player_number==2 else 1000
                self.pos_y= 300
                self.velocity =50



        def running(self,direction , width, height, other_player):
                if direction == None:
                        return
                x_change =0

                self.walkcount +=1
      
                if self.walkcount==9:
                        self.walkcount =0

                if direction == "left":
                        x_change -= self.velocity

                if direction == "right": 
                        x_change += self.velocity

                self.pos_x +=x_change

                if self.pos_x > width - self.player_width-150 or self.pos_x < 0:
                        self.pos_x -= x_change

                if other_player.pos_x - other_player.player_width< self.pos_x < other_player.pos_x + other_player.player_width:
                        self.pos_x -= x_change




class game:
        def __init__(self):
                self.P1 = player(2)
                self.P2 = player(1)

                self.screenwidth = 1600
                self.screenheight = 630
                self.gameDisplay = pygame.display.set_mode((self.screenwidth,self.screenheight))

                self.background = pygame.image.load('akuma.png')
                self.finish = False

                self.clock = pygame.time.Clock()


        def kick(self):

                
                self.gameDisplay.blit(self.P1.kick_images[self.P1.punch_kick_count], (self.P1.pos_x,self.P1.pos_y))
                if self.P1.pos_x + (self.P1.player_width + 80) > self.P2.pos_x:
                        self.gameDisplay.blit(self.P2.hurt_images[self.P1.punch_kick_count], (self.P2.pos_x,self.P2.pos_y))
                        return
                self.gameDisplay.blit(self.P2.character, (self.P2.pos_x,self.P2.pos_y))
                
        def punch(self):

                self.gameDisplay.blit(self.P1.punch_images[self.P1.punch_kick_count], (self.P1.pos_x,self.P1.pos_y))
                if self.P1.pos_x + (self.P1.player_width +80) > self.P2.pos_x:
                        self.gameDisplay.blit(self.P2.hurt_images[self.P1.punch_kick_count], (self.P2.pos_x,self.P2.pos_y))
                        return
                self.gameDisplay.blit(self.P2.character, (self.P2.pos_x,self.P2.pos_y))

        def left(self):

                x_change =0
                x_change += self.P1.velocity
                self.P1.pos_x -=x_change

                if self.P1.pos_x > self.screenwidth - self.P1.player_width-150 or self.P1.pos_x < 0:

                	self.P1.pos_x += x_change


                if self.P2.pos_x - self.P2.player_width< self.P1.pos_x < self.P2.pos_x + self.P2.player_width:

                    self.P1.pos_x += x_change

                self.gameDisplay.blit(self.P1.walk_left_images[self.P1.walkcount], (self.P1.pos_x,self.P1.pos_y))

                self.gameDisplay.blit(self.P2.character, (self.P2.pos_x,self.P2.pos_y))
 
        def right(self):


                x_change =0
                x_change += self.P1.velocity
                self.P1.pos_x +=x_change

                if self.P1.pos_x > self.screenwidth - self.P1.player_width-150 or self.P1.pos_x < 0:
                        self.P1.pos_x -= x_change

                if self.P2.pos_x -self.P2.player_width < self.P1.pos_x < self.P2.pos_x + self.P2.player_width:
                        self.P1.pos_x -= x_change
                self.gameDisplay.blit(self.P1.walk_right_images[self.P1.walkcount], (self.P1.pos_x,self.P1.pos_y))

                self.gameDisplay.blit(self.P2.character, (self.P2.pos_x,self.P2.pos_y))

        def position(self):


                self.gameDisplay.blit(self.P1.character, (self.P1.pos_x,self.P1.pos_y))
                self.gameDisplay.blit(self.P2.character, (self.P2.pos_x,self.P2.pos_y))



        def game_loop(self):
                centroid_old = 0
                centroid_new = 0

                # Mouse function

                while not self.finish:

                        _, frame = video.read()
                        (h, w) = frame.shape[:2]
                        test_image = cv2.resize(frame, (224, 224))
                        test_image = image.img_to_array(test_image)
                        test_image = np.expand_dims(test_image, axis = 0)
                        test_image = preprocess_input(test_image)
                        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)), 1.0, (300,300), (104.0,177.0,123.0))
                        net.setInput(blob)
                        faces = net.forward()


                        for i in range(0, faces.shape[2]):
                                confidence = faces[0,0,i,2]

                                if confidence < args["threshold"]:
                                    continue

                                box = faces[0,0,i,3:7] * np.array([w,h,w,h])
                                (startX, startY, endX, endY) = box.astype('int')

                                centroid_new = startX + endX

                        if centroid_new > centroid_old+30:
                                self.P1.walk_left = True
                                self.P1.walk_right = False

                        elif centroid_new < centroid_old - 30:
                                self.P1.walk_left = False
                                self.P1.walk_right = True
                        else:
                                self.P1.walk_left = False
                                self.P1.walk_right = False

                        centroid_old = centroid_new



                        for event in pygame.event.get():
                                pass



                        if model.predict(test_image).argmax() ==0 or self.P1.kick:
        
                                self.P1.kick = True

                                self.P1.punch_kick_count +=1

                                if self.P1.punch_kick_count ==4:
                                        self.P1.punch_kick_count =0
                                        self.P1.kick = False
                                        self.P1.punch = False


                                self.gameDisplay.blit(self.background, (0,0))
                                self.kick()
                                pygame.display.update()
                                self.clock.tick(60)
           


                        elif model.predict(test_image).argmax() ==2 or self.P1.punch :

                                self.P1.punch = True

                                self.P1.punch_kick_count +=1

                                if self.P1.punch_kick_count ==4:
                                        self.P1.punch_kick_count =0
                                        self.P1.punch = False
                                        self.P1.kick = False


                                self.gameDisplay.blit(self.background, (0,0))
                                self.punch()
                                pygame.display.update()
                                self.clock.tick(60)
                

                        elif self.P1.walk_left :
                                self.P1.walkcount +=1
                                if self.P1.walkcount ==9:
                                        self.P1.walkcount =0
                                        self.P1.punch = False
                                        self.P1.kick = False

                                self.gameDisplay.blit(self.background, (0,0))
                                self.left()
                                pygame.display.update()
                                self.clock.tick(60)


                        elif self.P1.walk_right :
                                self.P1.walkcount +=1
                                if self.P1.walkcount ==9:
                                        self.P1.walkcount =0
                                        self.P1.punch = False
                                        self.P1.kick = False

                                self.gameDisplay.blit(self.background, (0,0))
                                self.right()
                                pygame.display.update()
                                self.clock.tick(60)

                        else:
                                self.gameDisplay.blit(self.background, (0,0))
                                self.position()
                                pygame.display.update()
                                self.clock.tick(60)


                pygame.quit()
                quit()



##############################
game_play = game()
game_play.game_loop()
















