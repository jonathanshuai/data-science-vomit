import os
import json
from datetime import datetime

import numpy as np
import errno
import torchvision
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
'''
    TensorBoard Data will be stored in './runs' path
'''

class Parameters:
    """
    Example: params = Parameters(json_file)
    params.learning_rate
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

class Logger:
    def __init__(self, run_name, comment):
        self.run_name = run_name
        self.comment = comment

        # self.data_subdir = '{}_{}'.format(model_name, data_name)

        time_format = datetime.now().strftime('%h-%d-%Y-%Hh%Mm%Ss')
        self.directory = '{}/runs/{}/{}/'.format(os.getcwd(), self.run_name, time_format)
        # TensorBoard
        self.writer = SummaryWriter(self.directory, comment=self.comment)
        self.current_step = 1
        self.log_dir = '{}'

    def log(self, value, value_name):
        '''
        value    (int or float): Value to be recorded
        value_name        (str): Name of the value (for Tensorboard). 
        '''
        self.writer.add_scalar(
            '{}/{}'.format(self.run_name, value_name), value, self.current_step)

    def log_image(self, images, image_name, n_images=0, rows=4):
        '''
        images  (torch.Tensor): A tensor of images to write in the form of (BCHW)
        image_name       (str): Name of the image (for Tensorboard).
        n_images         (int): Number of images to show (takes top n_images from batch). Default all.
        rows             (int): Number of rows to display for grid. 
        '''

        if n_images == 0:
            n_images = images.shape[0]

        # Make a grid of images add it to the writer
        grid = torchvision.utils.make_grid(images[:n_images], normalize=True, scale_each=True, padding=0, nrow=rows)
        self.writer.add_image(image_name, grid, self.current_step)

    def step(self):
        '''
        Call this at the end of each iteration or time step.
        '''
        self.current_step += 1

    def get_step(self):
        return self.current_step

    def get_run_command(self):
        '''
        Returns the command to run (from the current directory)
        '''
        return 'tensorboard --logdir={} --host=127.0.0.1'.format(self.directory)

    def close(self):
        self.writer.close()