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
import torch
from pycontoursnake import Snake
from pycontoursnake.constraints import SpringConstraint
from pycontoursnake.shapes import Ellipse
import cv2
import numpy as np

image = cv2.imread('images/brain.png')
padding = 50

padded_image = np.pad(image, ((4*padding, 4*padding), (padding, padding), (0, 0)), mode='constant', constant_values=255)

height, width = padded_image.shape[:2]
center = (width // 2, height // 2)
#  Set parameters


N = 2000
shape_specs = {
    'center': center,
    'a': padded_image.shape[0] // 2.5,
    'b': padded_image.shape[1] // 3,
    'angle': 0.0,
}
initial_points = Ellipse().generate(N, **shape_specs)
def rest_length_fn(points):   
    distances = torch.linalg.norm(points - torch.tensor(center), axis=1)  
    mean_distance = torch.mean(distances)
    variance = torch.var(distances)
    return torch.pow(distances - mean_distance, 2) / variance

constraints = [
    SpringConstraint(pin_point=center, weight=0.00005, rest_length=rest_length_fn),
]

parameters = {
    'alpha': .05,
    'beta': 0.005,
    'sigma': 2,
    'gamma': 0.5,
    'delta': -0.5,
    'lr': 2,
    'verbose': True,
    'iter_max': 1000,
    'constraints': constraints,   
    'gradient_threshold': 0.10,
}



#  Initialize snake
snake = Snake(N, padded_image, points=initial_points, **parameters)
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
