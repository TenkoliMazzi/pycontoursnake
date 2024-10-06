import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
class Constraint:
    def __init__(self, weight, energy_fun : callable):
        self.weight = weight
        self.energy_fun = energy_fun

    def energy(self, points):                
        return self.weight * self.energy_fun(points)
    
    @staticmethod
    def spring(center,weight,rest_length):
        center = torch.tensor(center)

        if rest_length is callable:
            return Constraint(weight,lambda points: Constraint._spring_energy(center,points,rest_length(points)))
            

        return Constraint(weight,lambda points: Constraint._spring_energy(center,points,rest_length))
   
    @staticmethod
    def _spring_energy(center,points,rest_length):
        return torch.sum(((points - center.to(points.device)) - rest_length)**2) * 0.5
