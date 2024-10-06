import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
class Constraint:
    """
    The Constraint class represents a constraint with a specific weight and energy function.
    Attributes:
        weight (float): The weight of the constraint.
        energy_fun (callable): A function that calculates the energy of the constraint given a set of points.
    Methods:
        __init__(weight, energy_fun):
            Initializes the Constraint with a weight and an energy function.
        energy(points):
            Calculates the energy of the constraint for a given set of points.
              
    """
    def __init__(self, weight, energy_fun : callable):
        self.weight = weight
        self.energy_fun = energy_fun

    def energy(self, points):                
        return self.weight * self.energy_fun(points)
    
    @staticmethod
    def spring(center,weight,rest_length):
        """
        Creates a spring constraint with a specified center, weight, and rest length.
        -   center (tuple or torch.Tensor): The center of the spring.
        -   weight (float): The weight of the constraint.
        -   rest_length (float or callable): The rest length of the spring, or a function that calculates the rest length given a set of points."""
        center = torch.tensor(center)

        if rest_length is callable:
            return Constraint(weight,lambda points: Constraint._spring_energy(center,points,rest_length(points)))
            

        return Constraint(weight,lambda points: Constraint._spring_energy(center,points,rest_length))
   
    @staticmethod
    def _spring_energy(center,points,rest_length):
        return torch.sum(((points - center.to(points.device)) - rest_length)**2) * 0.5
