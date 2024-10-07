import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Shape:
    """
    The Shape class serves as a base class for different geometric shapes.
    """
    def generate(self, N, **shape_specs):
        raise NotImplementedError("This method should be overridden by subclasses")

class Star(Shape):
    """
    The Star class provides a method to generate star shapes.
    """
    def generate(self, N, **shape_specs):
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
        points_between_spikes = N // spikes
        cosine_spike_length = outer_radius - inner_radius
        def get_radius(i):
            angle = i * angle_step % total_angle
            in_spike_index = i % points_between_spikes
            return np.cos(2*in_spike_index / points_between_spikes * np.pi) * cosine_spike_length + inner_radius






        # Calculate the alternating radius pattern
        for i in range(N):
            # Calculate the current angle
            point_angle = torch.tensor(i * angle_step)
            
            # Use cosine around the circle for the alternating pattern    
            radius = get_radius(i)
            
            # Compute x and y positions for the point
            px = x + radius * torch.cos(point_angle)
            py = y + radius * torch.sin(point_angle)
            
            # Store the point in the tensor
            points[i, 0] = px
            points[i, 1] = py
        
        return points

class Ellipse(Shape):
    """
    The Ellipse class provides a method to generate ellipse shapes.
    """
    def generate(self, N, return_type = "torch", **shape_specs):
        """
        Generate an ellipse of 'N' points.
        - N (int): The number of points to generate for the ellipse.
        - shape_specs: Arbitrary keyword arguments for shape specifications:            
            - center (tuple): The (x, y) coordinates of the center of the ellipse. Default is (0, 0).
            - a (float): The major axis of the ellipse. Default is 200.
            - b (float): The minor axis of the ellipse. Default is 150.
            - angle (float): The angle of rotation of the ellipse in radians. Default is 0.           
        
        Returns:
        numpy.ndarray: An array of shape (N, 2) containing the (x, y) coordinates of the ellipse points.
        """
        # Extract shape specifications with default values
        x = shape_specs.get('center', (0, 0))[0]
        y = shape_specs.get('center', (0, 0))[1]
        a = shape_specs.get('a', 200)
        b = shape_specs.get('b', 150)
        if a < b:
            a, b = b, a
        angle = shape_specs.get('angle', 0)
        angle = torch.tensor(angle)
        # Tensor to hold the points
        points = torch.zeros(N, 2)

        # Total angular step between points
        total_angle = 2 * torch.pi

        for i in range(N):
            # Calculate the current angle
            point_angle = torch.tensor(i * total_angle / N)
            
            # Compute x and y positions for the point
            px = x + a * torch.cos(point_angle) * torch.cos(angle) - b * torch.sin(point_angle) * torch.sin(angle)
            py = y + a * torch.cos(point_angle) * torch.sin(angle) + b * torch.sin(point_angle) * torch.cos(angle)
            
            # Store the point in the tensor
            points[i, 0] = px
            points[i, 1] = py
            

        if return_type == "torch":
            return points
        else:
            return points.numpy()



      
        