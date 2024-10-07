import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pycontoursnake.utils import bcolors

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
    def __init__(self, weight, energy_fun: callable):
        self.weight = weight
        self.energy_fun = energy_fun

    def energy(self, points):
        return self.weight * self.energy_fun(points)


class SpringConstraint(Constraint):
    """
    The SpringConstraint class represents a spring constraint with a specific pin point, weight, and rest length.
    Attributes:
        pin_point (torch.Tensor): The fixed point of the spring.
        weight (float): The weight of the spring constraint.
        rest_length (float or callable): The rest length of the spring, or a function that calculates the rest length given a set of points.
        constrained_points (listlike): The indices of the points that are constrained by the spring.
    Methods:
        __init__(pin_point, weight, rest_length, constrained_points):
            Initializes the SpringConstraint with a pin point, weight, rest length, and constrained points.
        _spring_energy(center, points, rest_length, constrained_points):
            Calculates the spring energy for a given set of points.
    """
    def __init__(self, pin_point, weight, rest_length, constrained_points=-1):
        pin_point = torch.tensor(pin_point)
        self.pin_point = pin_point
        self.rest_length = rest_length
        self.constrained_points = constrained_points

        if callable(rest_length):
            energy_fun = lambda points: self._spring_energy(pin_point, points, rest_length(points), constrained_points)
        else:
            energy_fun = lambda points: self._spring_energy(pin_point, points, rest_length, constrained_points)

        super().__init__(weight, energy_fun)

    def __repr__(self):
        return f"SpringConstraint(pin_point={self.pin_point}, weight={self.weight}, rest_length={self.rest_length}, constrained_points={self.constrained_points})"

    @staticmethod
    def _spring_energy(center, points, rest_length, constrained_points):
    
        if constrained_points == -1:
            distance_from_center = points - center.to(points.device)
            displacement = distance_from_center - rest_length.unsqueeze(1)
            return torch.sum(displacement ** 2) * 0.5
        elif not isinstance(constrained_points, (list, tuple, torch.Tensor, np.ndarray)):
            print(bcolors.WARNING + "constrained_points must be a list, tuple, torch.Tensor, or numpy.ndarray")
        else:
            distance_from_center = points[constrained_points] - center.to(points.device)
            displacement = distance_from_center - rest_length[constrained_points].unsqueeze(1)
            return torch.sum(displacement ** 2) * 0.5
            

