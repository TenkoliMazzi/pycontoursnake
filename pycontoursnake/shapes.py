import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Shapes:

    @staticmethod
    def generate_star(N, **shape_specs):
        # Extract shape specifications with default values
        x = shape_specs.get('center', (0, 0))[0]
        y = shape_specs.get('center', (0, 0))[1]
        outer_radius = shape_specs.get('outer_radius', 100)
        inner_radius = shape_specs.get('inner_radius', 50)
        spikes = shape_specs.get('spikes', 5)
        
        # Tensor to hold the points
        points = torch.zeros((N, 2))
        
        # Total angular step between points
        total_angle = 2 * torch.pi
        angle_step = total_angle / N
        
        # Calculate the alternating radius pattern
        for i in range(N):
            # Calculate the current angle
            point_angle = torch.tensor(i * angle_step)
            
            # Determine whether this point is on an outer or inner radius
            if (i % (N // spikes)) < (N // (2 * spikes)):  # Spike region
                radius = outer_radius
            else:  # Inner region
                radius = inner_radius
            
            # Compute x and y positions for the point
            px = x + radius * torch.cos(point_angle)
            py = y + radius * torch.sin(point_angle)
            
            # Store the point in the tensor
            points[i, 0] = px
            points[i, 1] = py
        
        return points

    @staticmethod
    def generate_ellipse(N, image="", a=200, b=150, xo=0, yo=0):
        """Generate an ellipse of 'N' points."""
        if image == "":
            image = np.zeros((512, 512))
        a = a or image.shape[0] / 2
        b = b or image.shape[1] / 2
        xo = xo or image.shape[0] / 2
        yo = yo or image.shape[1] / 2

        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x = a * np.cos(theta) + xo
        y = b * np.sin(theta) + yo
        points = np.stack([x, y], axis=1)
        noise = np.random.normal(0, 10, points.shape)  # Reduced noise for smoother initialization
        return points