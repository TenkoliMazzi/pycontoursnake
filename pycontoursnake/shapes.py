import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Shapes:
    """
    The Shapes class provides static methods to generate geometric shapes such as stars and ellipses.   
    """

    @staticmethod
    def generate_star(N, **shape_specs):
        """
        Generates a star shape with specified parameters.
        -   N (int): The number of points to generate for the star.
        -   shape_specs: Arbitrary keyword arguments for shape specifications:
            - center (tuple): The (x, y) coordinates of the center of the star. Default is (0, 0).
            - outer_radius (float): The radius of the outer points of the star. Default is 100.
            - inner_radius (float): The radius of the inner points of the star. Default is 50.
            - spikes (int): The number of spikes in the star. Default is 5.
        Returns:
        torch.Tensor: A tensor of shape (N, 2) containing the (x, y) coordinates of the star points.
        """
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
        """
        Generate an ellipse of 'N' points.
        - N (int): The number of points to generate for the ellipse.
        - image (numpy.ndarray): The image to use as reference for the ellipse size. Default is an empty image.
        - a (int): The major axis of the ellipse. Default is half the image width.
        - b (int): The minor axis of the ellipse. Default is half the image height.
        - xo (int): The x-coordinate of the ellipse center. Default is half the image width.
        - yo (int): The y-coordinate of the ellipse center. Default is half the image height.
        Returns:
        numpy.ndarray: An array of shape (N, 2) containing the (x, y) coordinates of the ellipse points.
        """
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