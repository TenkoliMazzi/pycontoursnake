
from matplotlib.animation import FuncAnimation
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pycontoursnake.shapes import Ellipse


        
class Snake:
    """
    Class to represent a snake contour in an image.
    
    The snake is represented as a set of points in a 2D image and is optimized using Gradient Descent to minimize energy.
    Energy is defined as a combination of internal and external forces.
    
    Internal forces include:
    - Elasticity
    - Curvature
    
    External forces include:
    - Image gradient
    - Intensity
    
    The snake is initialized as an ellipse if no points are provided. Parameters can be provided as keyword arguments or set manually.
    
    Keyword arguments:
    - N: Number of points in the snake contour.
    - image: 3-channel color image to segment.
    - points: Initial points for the snake contour.
    - alpha: Elasticity weight.
    - beta: Curvature weight.
    - sigma: Gaussian blur standard deviation for image gradient.
    - gamma: External energy weight.
    - delta: Image intensity energy weight.
    - lr: Learning rate for gradient descent.
    - verbose: Print energy values at each iteration.
    - iter_max: Maximum iterations for snake optimization.
    - gradient_threshold: Treshold for the gradient magnitude.
    
    Note:
    - If both points and N are provided, N will be set to the number of points.
    - If both a keyword argument and manual parameter setting are used, the keyword argument will take precedence.
    - If points are provided, they should be in the form [(x1, y1), (x2, y2), ...].
    - Image should be a 3-channel color image.
    
    METHODS:
    - compute_energy(): Compute total energy for the current snake configuration.
    - compute_internal_energy(): Compute internal energy for elasticity and curvature.
    - compute_external_energy(): Compute external energy from image gradients and intensity.
    - compute_sobel(sigma): Precompute Sobel gradient and magnitude.
    - gradient_descent(): Run optimization using gradient descent.
    - plot_snake(): Plot the current snake on the image.
    """
    
    def __init__(self, N, image, points=[], alpha=0.1, beta=0.1, sigma=20, gamma=2, delta=2, lr=0.01, verbose=False, iter_max=100,batch_size = .75, constraints = [], gradient_threshold = 0, **kwargs):
        self.device = torch.device("cpu")
        # If points are provided, use them, otherwise generate an ellipse
        

        self.alpha = alpha  # Elasticity weight
        self.beta = beta  # Curvature weight
        self.sigma = sigma  # Gaussian blur standard deviation
        self.gamma = gamma  # External energy weight
        self.delta = delta  # Image intensity energy weight
        self.lr = lr  # Learning rate for gradient descent
        self.batch_size = batch_size  # Batch size for optimization
        self.iter_max = iter_max  # Max iterations for optimization
        self.constraints = constraints  # Constraints for the snake
        self.verbose = verbose
        self.N = N  # Number of points
        self.gradient_threshold = gradient_threshold  # Threshold for gradient magnitude

        if kwargs is not None:
            for key, value in kwargs.items():
                print(f"Setting {key} to {value}")
                setattr(self, key, value)

        if image is torch.Tensor:
            self.image_numpy = cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_BGR2GRAY)
        else:
            self.image_numpy = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        self.image_tensor = torch.tensor(self.image_numpy, dtype=torch.float32)
        self.original_image_tensor = self.image_tensor.clone().detach()   
       
        self.compute_sobel(sigma=self.sigma)  # Precompute Sobel gradient
        
        
        
        if len(points) > 0:
            self.points = points.clone().detach().to(self.device).requires_grad_(True) if torch.is_tensor(points) else torch.tensor(points, dtype=torch.float32, requires_grad=True).to(self.device)
            N = len(points)  # Ensure N matches the number of points
        else:
            ellipse_parameters = {
                'center': (self.image_tensor.shape[1] // 2, self.image_tensor.shape[0] // 2),
                'a': self.image_tensor.shape[1] // 4,
                'b': self.image_tensor.shape[0] // 4
            }
            self.points = torch.tensor(Ellipse().generate(N,ellipse_parameters), dtype=torch.float32, requires_grad=True).to(self.device)

        self.displacement = torch.sqrt(torch.sum((self.points - self.points.roll(shifts=1, dims=0))**2, dim=1))
        self.curvature = torch.sqrt(torch.sum((self.points - 2 * self.points.roll(shifts=1, dims=0) + self.points.roll(shifts=2, dims=0))**2, dim=1))
 
    def compute_energy(self):
        """Compute the total energy for the snake."""
        internal_energy = self.compute_internal_energy()
        external_energy = self.compute_external_energy()
        constraint_energy = self.compute_constraint_energy()
        return external_energy + internal_energy + constraint_energy

    def compute_internal_energy(self):
        """Compute the internal energy (elasticity and curvature)."""
        self.displacements = (self.points - self.points.roll(shifts=1, dims=0))
        self.curvatures = (self.points - 2 * self.points.roll(shifts=1, dims=0) + self.points.roll(shifts=2, dims=0))
        self.rest_length = self.displacements.mean()
        self.rest_curvature = self.curvatures.mean()
        displacement_energy = torch.sum((self.displacements - self.rest_length)**2) * 0.5
        curvature_energy = torch.sum((self.curvatures - self.rest_curvature)**2) * 0.5        
        return self.alpha * displacement_energy + self.beta * curvature_energy   
 
    

    def compute_curvature(self, point, prev_point, next_point):
        """Compute the curvature  for a single point in the snake."""
        return ((point - 2 * prev_point + next_point) - self.rest_curvature) / 2
        
   
    def compute_displacement(self, prev_point, next_point):
        """Compute the displacement using central differences."""
        return ((next_point - prev_point) - self.rest_length) / 2
    


    def compute_external_energy(self):
        """Compute the external energy using bilinear interpolation for differentiability."""
        
        # Get the x and y coordinates of the points
        x_points = self.points[:, 0]
        y_points = self.points[:, 1]
        
        # Create a grid of points normalized to [-1, 1]
        height, width = self.gradient_magnitude.shape
        grid_x = (2 * x_points / (width - 1)) - 1  # Normalize x to [-1, 1]
        grid_y = (2 * y_points / (height - 1)) - 1  # Normalize y to [-1, 1]

        # Stack and form the sampling grid
        grid = torch.stack((grid_x, grid_y), dim=1).unsqueeze(0).unsqueeze(0).double().to(self.device)

        # Bilinear interpolation to sample gradient magnitude and image tensor
        gradient_energy = F.grid_sample(self.gradient_magnitude.unsqueeze(0).unsqueeze(0).double(), grid, align_corners=True)
        intensity_energy = F.grid_sample(self.original_image_tensor.unsqueeze(0).unsqueeze(0).double(), grid, align_corners=True) / 255.0

        # Compute the total energy
        total_gradient_energy = -torch.sum(gradient_energy ** 2)
        total_intensity_energy = torch.sum(intensity_energy ** 2)
        
        total_external_energy = self.gamma * total_gradient_energy + self.delta * total_intensity_energy

        return total_external_energy
        
    def compute_constraint_energy(self):
        """Compute the energy from constraints"""



        constraint_tensor = torch.stack([constraint.energy(self.points) for constraint in self.constraints])
        return torch.sum(constraint_tensor)
        

    def compute_sobel(self, sigma=50):
        """Precompute Sobel gradient and magnitude after Gaussian smoothing."""
        
        self.image_numpy = cv2.GaussianBlur(self.image_numpy, (0, 0), sigma)
        self.image_tensor = torch.tensor(self.image_numpy, dtype=torch.float32)

        sobel_x = cv2.Sobel(self.image_numpy, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.image_numpy, cv2.CV_64F, 0, 1, ksize=3)

        self.gradient = torch.stack((torch.tensor(sobel_x), torch.tensor(sobel_y)), dim=2).to(self.device)
        self.gradient_magnitude = torch.sqrt(self.gradient[:, :, 0]**2 + self.gradient[:, :, 1]**2)
        self.gradient_magnitude = (self.gradient_magnitude - self.gradient_magnitude.min()) / (self.gradient_magnitude.max() - self.gradient_magnitude.min())
        self.gradient_magnitude = torch.where(self.gradient_magnitude < self.gradient_threshold, 0, self.gradient_magnitude)

    def gradient_descent(self, plot_skip=-1):
        """Run gradient descent to optimize the snake."""
        self.old_points = self.points.clone().detach().unsqueeze(2)
        optimizer = torch.optim.Adam([self.points], lr=self.lr)
        for i in range(self.iter_max):
            optimizer.zero_grad()
            energy = self.compute_energy()
            energy.backward()
            optimizer.step()
            # record the points in a tensor [N,2,iter_max]
            self.old_points = torch.cat((self.old_points, self.points.clone().detach().unsqueeze(2)), dim=2)
        
            if self.verbose and i % plot_skip == 0:
                print(f"Iteration {i}: Energy = {energy.item()}")
                print(f"Inernal Energy: {self.compute_internal_energy().item()} External Energy: {self.compute_external_energy().item()} Constraint Energy: {self.compute_constraint_energy().item()}")

            if plot_skip != -1 and i % plot_skip == 0:
                self.plot_snake()       

        self.plot_snake()       
    def plot_snake(self):
        """Plot the current snake on the image."""
        points_and_first_x = torch.cat((self.points[:, 0], self.points[0, 0].unsqueeze(0))).detach().cpu().numpy()
        points_and_first_y = torch.cat((self.points[:, 1], self.points[0, 1].unsqueeze(0))).detach().cpu().numpy()
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(self.gradient_magnitude.cpu(), cmap='gray')
        plt.plot(points_and_first_x, points_and_first_y, 'r', lw=1, label='Snake')
        plt.subplot(1, 2, 2)
        plt.imshow(self.original_image_tensor, cmap='gray')
        plt.plot(points_and_first_x, points_and_first_y, 'r', lw=1, label='Snake')
        plt.show()

    def plot_evolution(self):
        """Plot the evolution of the snake over iterations."""
        # Creo un'animazione che mostra l'evoluzione dei punti nel tempo
        points = self.old_points.cpu().detach().numpy()

        fig, ax = plt.subplots()
        ax.imshow(self.original_image_tensor, cmap='gray')

        # Imposta la linea iniziale, vuota all'inizio
        line, = ax.plot([], [], 'r', lw=1, label='Snake')

        # Imposta i limiti degli assi
        ax.set_xlim(0, self.original_image_tensor.shape[1])
        ax.set_ylim(self.original_image_tensor.shape[0], 0)
        ax.set_title('Snake Evolution')
        ax.legend()

        # Funzione per inizializzare l'animazione
        def init():
            line.set_data([], [])
            return line,

        # Funzione per aggiornare l'animazione ad ogni frame
        def update(frame):
            line.set_data(points[:, 0, frame], points[:, 1, frame])        
            ax.set_title(f'Snake Evolution - Iteration {frame}')    
            return line,

        # Crea l'animazione
        ani = FuncAnimation(fig, update, frames=range(self.iter_max), init_func=init, blit=False, interval=1)  # intervallo 200ms per frame


        plt.show()

        return ani




    