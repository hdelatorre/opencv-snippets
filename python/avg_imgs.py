import cv2
import time
import os
import datetime
import sys
import numpy as np

    
class ImageAverager(object):
    def __init__(self, frames_to_avg=5):
        dir_path = "./"

        now = int(time.time())
        fileName = 'avg{}.png'.format(str(now))

        camera_port = 0
        ramp_frames = 120
        keep_frames = 5

        camera = cv2.VideoCapture(camera_port)

        camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1024)

        summed_img = None

        # loop until enter is pressed
        while not self.was_enter_pressed():
            # get the number of frames to average from the camera
            for i in xrange(frames_to_avg):
                retval, img = camera.read()

                # print the largest pixel value in the image
                print 'input image #{} max: {}'.format(i, img.max())
                # if we are just starting,
                # create a new image which can hold all the summing we do
                if summed_img is None:
                    summed_img = np.zeros((
                                           img.shape[0],
                                           img.shape[1],
                                           3
                                          ),
                                          dtype = np.uint32)
                summed_img += img
            avg_img = summed_img / float(frames_to_avg)
            
            print 'average image max: {}'.format(avg_img.max())
            avg_img = np.array(avg_img, dtype = np.uint8)

            cv2.imshow('{} images averaged'.format(frames_to_avg), avg_img)
            cv2.imshow('last input image', img)
            
            # reset the intermediate image
            summed_img = None

        # shut off the camera
        del(camera)

        # write out the last averaged image
        cv2.imwrite(dir_path + fileName, avg_img)

    def was_enter_pressed(self):
        # wait for enter to be pressed
        self.key_code = cv2.waitKey()

        # just look at the LSB, since cv2.waitKey() seems to
        # return different values on differernt platforms
        print 'key_code: {}'.format(self.key_code%256)
        
        #try converting the int to a character
        self.key_char = chr(self.key_code%256) if self.key_code%256 < 128 else '?'
        print 'key_char: {}'.format(self.key_char)
        
        # check if enter was pressed
        return  self.key_char == '\n'
           
if __name__=='__main__':
    ImageAverager()