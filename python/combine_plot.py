#!/usr/bin/env python

'''
Add or multiply two graph lines of different color using their Y values (in image coordinates).

USAGE
  combine_plots.py <input_image> <output_image>

  With mouse select any point on the graph where Y=0 then press enter.
  With mouse select any point that matches the color of the first plot line.
  With mouse select any point that matches the color of the second plot line.
'''
from __future__ import print_function
import numpy as np
import cv2
import os
import math
import PythonMagick


class combine_plots(object):
    def __init__(self, input_image_path, added_output_image_path, multiplied_output_image_path):
        dir_path, image_filename = os.path.split(input_image_path)
        self.added_output_image_path = added_output_image_path
        self.multiplied_output_image_path = multiplied_output_image_path
        self.dir_path = dir_path
        self.image_filename, self.image_ext = os.path.splitext(image_filename)

        if input_image_path.lower().endswith('.gif'):
            input_image_path = self.convert_image_to_png(
                                   dir_path,
                                   self.image_filename + self.image_ext, 
                                   self.image_filename + '.png')
        print(input_image_path)
        self.img = cv2.imread(input_image_path)
        self.img_win = 'show_and_select_color'
        
    def start(self):
        self.show_and_select_color()
        self.show_thresh_image()
        self.add_and_multiply_and_scale_plots()
        self.save_imgs()

    def save_imgs(self):
        cv2.imwrite(self.added_output_image_path, self.added)
        cv2.imwrite(self.multiplied_output_image_path, self.multiplied)

        a2 = os.path.splitext(self.added_output_image_path)[0] + '_scaled.png'
        m2 = os.path.splitext(self.multiplied_output_image_path)[0] + '_scaled.png'
        merged = os.path.splitext(self.multiplied_output_image_path)[0] + '_merged.png'

        a2_img = cv2.resize(self.added, dsize=None, fx=1, fy=0.25, interpolation = cv2.INTER_CUBIC)
        m2_img = cv2.resize(self.multiplied, dsize=None, fx=1, fy=0.5, interpolation = cv2.INTER_CUBIC)

        cv2.imwrite(a2, a2_img)
        cv2.imwrite(m2, m2_img)
        cv2.imwrite(merged, self.combined)

    def show_and_select_color(self):
        self.color = None

        cv2.imshow(self.img_win, self.img)
        cv2.setMouseCallback(self.img_win, self.onmouse)

        print('Select the plot baseline, then press ENTER when satisfied.')
        self.wait_for_enter_press()
        self.baseline_x = self.x
        self.baseline_y = self.y
        print('\nBaseline Y value selected to be: ({},{})'.format(self.x, self.y))

        print('\n\nSelect the first plot line color, then press ENTER when satisfied.')
        self.wait_for_enter_press()
        self.plot1_color = self.color
        print('\nFirst plot color selected to be: ' + str(self.color))
        
        print('\n\nSelect the second plot line color, then press ENTER when satisfied.')
        self.wait_for_enter_press()
        self.plot2_color = self.color
        print('\nSecond plot color selected to be: ' + str(self.color))

    def show_thresh_image(self):
        self.plot1_img = self.get_filtered_plot_image(self.img, self.plot1_color)
        self.plot2_img = self.get_filtered_plot_image(self.img, self.plot2_color)
        cv2.imshow('Plot 1 output window', self.plot1_img)
        cv2.imshow('Plot 2 output window', self.plot2_img)
        self.wait_for_enter_press()

    def add_and_multiply_and_scale_plots(self):
        height = self.plot1_img.shape[0]
        width = self.plot1_img.shape[1]

        plot1_ys, plot1_xs = np.where(self.plot1_img==255)
        plot2_ys, plot2_xs = np.where(self.plot2_img==255)

        added = np.zeros((height*2, width, 3), dtype = self.img.dtype)
        multiplied =  np.zeros((height*1.1, width, 3), dtype = self.img.dtype)
        self.combined =  np.zeros((height*1.1, width, 3), dtype = self.img.dtype)
        for x in xrange(0, width):
            if (x in plot1_xs) and (x in plot2_xs):
                ys1 = plot1_ys[plot1_xs == x]
                ys2 = plot2_ys[plot2_xs == x]
                added_y = max(ys1) + max(ys2)
                multiplied_y = max(ys1) * max(ys2)
                added[added_y , x] = [51, 242, 245]
                # uncomment the following lines to show the original plot lines from the source image
                #added[ys1-height, x] = self.img[ys1, x]
                #added[ys2-height, x] = self.img[ys2, x]

                multiplied[(multiplied_y/(height))-(height) , x] = [96, 51, 235]
                # uncomment the following lines to show the original plot lines from the source image
                #multiplied[ys1-height, x] = self.img[ys1, x]
                #multiplied[ys2-height, x] = self.img[ys2, x]

                self.combined[(multiplied_y/(height*2))-(height/2) , x] = [96, 51, 235]
                self.combined[(added_y/4) - (height/2), x] = [51, 242, 245]

        cv2.imshow('Plots added', added)
        self.wait_for_enter_press()
        self.added = added
        self.multiplied = multiplied

    def get_filtered_plot_image(self, orig, plot_color):
        b_lo = max(plot_color[0]-(64), 0)
        b_hi = min(plot_color[0]+(64), 255)
        g_lo = max(plot_color[1]-(64), 0)
        g_hi = min(plot_color[1]+(64), 255)
        r_lo = max(plot_color[2]-(64), 0)
        r_hi = min(plot_color[2]+(64), 255)
        return cv2.inRange(orig, (b_lo, g_lo, r_lo), (b_hi, g_hi, r_hi))

    def convert_image_to_png(self, basedir, src, dest):
        image = PythonMagick.Image(basedir + '/' + src)
        image.write(os.path.join(basedir, dest))
        return (os.path.join(basedir, dest))

    def onmouse(self, event, x, y, flags, param):
        if flags & cv2.EVENT_FLAG_LBUTTON:
            if self.color is None:
                # set up a small window that shows the currently selected color
                self.img_color_selected = np.zeros((64,64,3), np.uint8)
                self.img_color_selected_win = 'Color Currently Selected'
                cv2.imshow(self.img_color_selected_win,
                       self.img_color_selected)
            self.color = self.img[y,x]
            self.x = x
            self.y = y
            # update the currently selected color image with the new color
            self.img_color_selected[:,:] = self.color
            # show the new currently selected color image
            cv2.imshow(self.img_color_selected_win,
                       self.img_color_selected)
            print('Just clicked (x,y):({},{}) with color: {}'.format(self.x, self.y, self.color))

    def wait_for_enter_press(self):
        # wait for enter to be pressed
        key_pressed = cv2.waitKey()
        #print(key_pressed)
        while key_pressed != 10:
            key_pressed = cv2.waitKey()
            print(key_pressed)
        key_pressed = None
        

if __name__ == '__main__':
    print(__doc__)

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '')
    if len(args)==3:
        input_path = args[0]
        added_output_path = args[1]
        multiplied_output_path = args[2]

        combiner = combine_plots(os.path.abspath(input_path), os.path.abspath(added_output_path), os.path.abspath(multiplied_output_path))
        combiner.start()
    else:
        # I used this for debugging, since I didn't want to choose colors again
        input_path = '../imgs/multiline_plot_cropped.png'
        added_output_path = './combine_plot_out/xte1_added.png'
        multiplied_output_path = './combine_plot_out/xte1_multiplied.png'

        combiner = combine_plots(os.path.abspath(input_path), os.path.abspath(added_output_path), os.path.abspath(multiplied_output_path))
        combiner.baseline_y = 353
        
        combiner.plot1_color = [204, 85, 0]
        combiner.plot2_color = [0, 0, 204]
        combiner.show_thresh_image()
        combiner.add_and_multiply_and_scale_plots()
        combiner.save_imgs()
    
    cv2.destroyAllWindows()
