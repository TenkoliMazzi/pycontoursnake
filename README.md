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

This example uses the pysnakecontour package to initialize a snake around a star-shaped object and optimize its contour to fit the object's boundary.

```python
from pysnakecontour import Snake, Shapes, Constraint
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib ipympl

# Load an example image or create a blank image with a white star in the middle
canvas_shape = (1000, 1000, 3)
shape_specs = {
    'center': (500, 500),
    'radius': 150,
    'inner_radius': 150,
    'outer_radius': 250,
    'spikes': 4,
    'color': (255, 255, 255),
    'thickness': -1,
    'ratio': 2,
    'noise': 0.0,  # No noise
}

# Create a blank image with the star shape
star_on_white_background = np.zeros(canvas_shape, dtype=np.int32) * 255
star_points = Shapes.generate_star(N=50, **shape_specs).int()
star_on_white_background = cv2.fillPoly(star_on_white_background, pts=[star_points.numpy()], color=(255, 255, 255))
star_on_white_background = cv2.convertScaleAbs(star_on_white_background)

# Generate initial points in an ellipse around the center of the star
initial_points = Shapes.generate_ellipse(N=50, xo=500, yo=500, a=400, b=400)

# Define constraints and parameters
constraints = [Constraint.spring([500, 500], 0.00005, 100)]

parameters = {
    'alpha': 0.0005,   # Elasticity weight
    'beta': 0.0005,    # Curvature weight
    'gamma': 0.05,     # Gradient energy weight
    'delta': 0.25,     # Intensity energy weight
    'sigma': 10,       # Gaussian blur standard deviation
    'step_size': 0.02, # Step size for finer adjustments
    'lr': 0.05,        # Learning rate
    'verbose': False,  # Disable verbose output
    'constraints': constraints
}

# Initialize the Snake algorithm
snake = Snake(
    N=20000,  # Number of control points for the snake
    image=star_on_white_background,  # Input image
    points=initial_points,  # Initial snake points (ellipse)
    iter_max=5000,  # Maximum iterations for optimization
    **parameters  # Additional parameters
)

# Run gradient descent to optimize the snake's position
snake.gradient_descent(plot_skip=-1)  # plot_skip=-1 disables intermediate plotting

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
![Snake Evolution](docs/snake_example.gif)

## License
This project is licensed under the MIT License.
