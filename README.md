# Snake Contour Optimization with PyTorch

This repository provides an implementation of an active contour (or "snake") algorithm for segmenting 2D images using gradient descent. The snake contour evolves to minimize energy based on internal and external forces. The internal forces enforce smoothness, while external forces drive the contour towards features in the image, such as edges.

## Features
- **Internal energy**: Enforces smoothness through elasticity and curvature terms.
- **External energy**: Attracts the snake towards edges and image intensity features.
- **Gradient-based optimization**: Snake evolution is optimized using gradient descent with backpropagation.
- **Flexible initialization**: You can initialize the snake with a set of points or automatically with an ellipse.
- **Visualization**: Real-time snake evolution animation during optimization.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- NumPy
- Matplotlib

You can install the required dependencies using:

```bash
pip install torch opencv-python-headless numpy matplotlib
```

## Example Usage

This example uses the pysnakecontour package to initialize a snake around a brain.
```python
import torch  # Import the PyTorch library
from pycontoursnake import Snake  # Import the Snake class from the pycontoursnake package
from pycontoursnake.constraints import SpringConstraint  # Import the SpringConstraint class
from pycontoursnake.shapes import Ellipse  # Import the Ellipse class
import cv2  # Import the OpenCV library
import numpy as np  # Import the NumPy library

# Read the image from the specified path
image = cv2.imread('images/brain.png')
padding = 50  # Define padding size

# Pad the image with a constant value (255) to avoid boundary issues
padded_image = np.pad(image, ((4*padding, 4*padding), (padding, padding), (0, 0)), mode='constant', constant_values=255)

# Get the height and width of the padded image
height, width = padded_image.shape[:2]
center = (width // 2, height // 2)  # Calculate the center of the image

# Set parameters for the initial snake shape
N = 1000  # Number of points in the snake
shape_specs = {
    'center': center,  # Center of the ellipse
    'a': padded_image.shape[0] // 2.5,  # Semi-major axis length
    'b': padded_image.shape[1] // 3,  # Semi-minor axis length
    'angle': 0.0,  # Rotation angle of the ellipse
}

# Generate initial points for the snake in the shape of an ellipse
initial_points = Ellipse().generate(N, **shape_specs)

# Define a function to calculate the rest length for the spring constraint
def rest_length_fn(points):   
    distances = torch.linalg.norm(points - torch.tensor(center), axis=1)  # Calculate distances from center
    mean_distance = torch.mean(distances)  # Calculate mean distance
    variance = torch.var(distances)  # Calculate variance of distances
    return torch.pow(distances - mean_distance, 2) / variance  # Return normalized squared distances

# Define constraints for the snake
constraints = [
    SpringConstraint(pin_point=center, weight=0.00005, rest_length=rest_length_fn),  # Add a spring constraint
]

# Set parameters for the snake optimization
parameters = {
    'alpha': .05,  # Elasticity weight
    'beta': 0.005,  # Curvature weight
    'sigma': 2,  # Gaussian smoothing parameter
    'gamma': 0.5,  # Gradient energy weight
    'delta': -0.5,  # Intensity energy weight
    'lr': 2,  # Learning rate
    'verbose': True,  # Verbose output
    'iter_max': 500,  # Maximum number of iterations
    'constraints': constraints,  # Constraints to apply
    'gradient_threshold': 0.10,  # Gradient threshold
}

# Initialize the snake with the specified parameters and image
snake = Snake(N, padded_image, points=initial_points, **parameters)
# Perform gradient descent to evolve the snake
snake.gradient_descent(plot_skip=250)
```

### Key Parameters:
- **`N`**: Number of snake points.
- **`alpha`**: Elasticity weight (controls how much the snake stretches).
- **`beta`**: Curvature weight (controls how much the snake bends).
- **`gamma`**: Gradient energy weight (attracts the snake to image edges).
- **`delta`**: Intensity energy weight (attracts the snake to regions of high intensity).
- **`lr`**: Learning rate for gradient descent.
- **`iter_max`**: Maximum number of optimization iterations.

### Visualization
- **`plot_snake()`**: Plots the snake's current position on the image.
- **`plot_evolution()`**: Shows an animated evolution of the snake contour over the optimization process.

## Example
![Snake Evolution](examples/images/brain.gif)

## License
This project is licensed under the MIT License.
