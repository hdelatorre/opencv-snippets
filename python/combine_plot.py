#!/usr/bin/env python

'''
Add or multiply two graph lines of different color using their Y values (in image coordinates).

USAGE
  combine_plots.py <input image path> <output dir path> <num plot lines to extract>

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
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator


class combine_plots(object):
    def __init__(self, input_image_path, output_image_dir_path, num_plots_to_extract):
        dir_path, image_filename = os.path.split(input_image_path)
        self.input_image_path = input_image_path
        self.output_image_dir_path = output_image_dir_path
        if not os.path.isdir(self.output_image_dir_path):
            raise Exception('Output directory does not exist, please create it. ({})'.format(self.output_image_dir_path))
        #self.added_output_image_path = added_output_image_path
        #self.multiplied_output_image_path = multiplied_output_image_path
        self.dir_path = dir_path
        self.image_filename, self.image_ext = os.path.splitext(image_filename)

        self.num_plots_to_extract = num_plots_to_extract
        self.plot_colors = []
        self.extracted_plot_imgs = []

        if input_image_path.lower().endswith('.gif'):
            input_image_path = self.convert_image_to_png(
                                   dir_path,
                                   self.image_filename + self.image_ext, 
                                   self.image_filename + '.png')
        #print(input_image_path)
        self.img = cv2.imread(input_image_path)
        self.img_win = 'show_and_select_color'
        
    def start(self):
        self.show_and_select_color()
        self.show_thresh_image()
        self.add_and_multiply_and_scale_plots()
        self.save_imgs()

    def save_imgs(self):
        img_name = os.path.splitext(os.path.split(self.input_image_path)[1])[0]
        output_img_prefix = os.path.join(self.output_image_dir_path, img_name)

        added_path        = output_img_prefix + '_added.png'
        added_scaled_path = output_img_prefix + '_added_scaled.png'
        multiplied_path   = output_img_prefix + '_multiplied.png'
        merged_path       = output_img_prefix + '_multiplied_plus_added.png'

        cv2.imwrite(added_path, self.added)
        cv2.imwrite(added_scaled_path, self.added_scaled)
        cv2.imwrite(multiplied_path, self.multiplied)
        cv2.imwrite(merged_path, self.combined)

        for i in xrange(0, self.num_plots_to_extract):
            file_name = output_img_prefix + '_extracted_{}.png'.format(i)
            cv2.imwrite(file_name, self.extracted_plot_imgs[i])
        ##a2_img = cv2.resize(self.added, dsize=None, fx=1, fy=0.25, interpolation = cv2.INTER_CUBIC)
        ##m2_img = cv2.resize(self.multiplied, dsize=None, fx=1, fy=0.5, interpolation = cv2.INTER_CUBIC)

    def show_and_select_color(self):
        self.color = None

        cv2.imshow(self.img_win, self.img)
        cv2.setMouseCallback(self.img_win, self.onmouse)

        print('Select the plot baseline, then press ENTER when satisfied.')
        self.wait_for_enter_press()
        self.baseline_x = self.x
        self.baseline_y = self.y
        print('\nBaseline Y value selected to be: ({},{})'.format(self.x, self.y))

        for img_num in xrange(0, self.num_plots_to_extract):
            print('\n\nSelect the #{} plot line color, then press ENTER when satisfied.'.format(img_num))
            self.wait_for_enter_press()
            self.plot_colors.append(self.color)
            print('#{} plot color selected to be: '.format(img_num) + str(self.color))
        

    def show_thresh_image(self):
        for img_num in xrange(0, self.num_plots_to_extract):
            self.extracted_plot_imgs.append(self.get_filtered_plot_image(self.img, self.plot_colors[img_num]))
            cv2.imshow('Plot {} output window'.format(img_num), self.extracted_plot_imgs[-1])
        self.wait_for_enter_press()

    def add_and_multiply_and_scale_plots(self):
        height = self.img.shape[0]
        width = self.img.shape[1]

        extracted_plot_ys = []
        extracted_plot_xs = []
        for img in self.extracted_plot_imgs:        
            plot_ys, plot_xs = np.where(img==255)
            extracted_plot_ys.append(plot_ys)
            extracted_plot_xs.append(plot_xs)

        added = np.zeros((height*self.num_plots_to_extract, width, 3), dtype = self.img.dtype)
        multiplied =  np.zeros((height*1.1, width, 3), dtype = self.img.dtype)
        self.combined =  np.zeros((height*1.1, width, 3), dtype = self.img.dtype)
        self.added_scaled = self.combined.copy()
        for x in xrange(0, width):
            if all(x in plot_xs for plot_xs in extracted_plot_xs):
                extracted_ys = [plot_ys[extracted_plot_xs[i] == x] for i, plot_ys in enumerate(extracted_plot_ys)]
                
                #added_y = max(ys1) + max(ys2)
                added_y = sum([max(ys) for ys in extracted_ys])
                #multiplied_y = max(ys1) * max(ys2)
                multiplied_y = reduce(operator.mul, [max(ys) for ys in extracted_ys])
                multiplied_y_scaled = multiplied_y**(1./self.num_plots_to_extract)
                added[added_y , x] = [51, 242, 245]
                self.added_scaled[float(added_y)/self.num_plots_to_extract, x] = [51, 242, 245]
                # uncomment the following lines to show the original plot lines from the source image
                #added[ys1-height, x] = self.img[ys1, x]
                #added[ys2-height, x] = self.img[ys2, x]

                multiplied[multiplied_y_scaled, x] = [96, 51, 235]
                # uncomment the following lines to show the original plot lines from the source image
                #multiplied[ys1-height, x] = self.img[ys1, x]
                #multiplied[ys2-height, x] = self.img[ys2, x]

                self.combined[multiplied_y_scaled, x] = [96, 51, 235]
                self.combined[float(added_y)/self.num_plots_to_extract, x] = [51, 242, 245]

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
        output_image_dir_path = args[1]
        #added_output_path = args[1]
        #multiplied_output_path = args[2]
        num_plots_to_extract = int(args[2])

        combiner = combine_plots(os.path.abspath(input_path), os.path.abspath(output_image_dir_path), num_plots_to_extract)
        combiner.start()
    else:
        # I used this for debugging, since I didn't want to choose colors again
        #input_path = '../imgs/multiline_plot_cropped.png'
        #added_output_path = './combine_plot_out/xte1_added.png'
        #multiplied_output_path = './combine_plot_out/xte1_multiplied.png'

        input_path = '/home/nathan/Downloads/xbd1.GIF'
        #added_output_path = './xbd1_added.png'
        #multiplied_output_path = './xbd1_multiplied.png'
        output_image_dir_path = "./out"

        combiner = combine_plots(os.path.abspath(input_path), os.path.abspath(output_image_dir_path), 3)
        combiner.baseline_y = 353
        
        combiner.plot_colors = [ [204, 85, 0], [0, 0, 204], [102, 170 ,  0] ]
        
        combiner.show_thresh_image()
        combiner.add_and_multiply_and_scale_plots()
        combiner.save_imgs()
    
    cv2.destroyAllWindows()
