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

## Usage

To create a snake and optimize it on a given image:

```python
import cv2
from snake import Snake

# Load image and convert to 3 channels if necessary
image = cv2.imread('path_to_image.jpg')

# Initialize the snake with parameters (optional arguments available)
snake = Snake(N=100, image=image, alpha=0.1, beta=0.1, gamma=2, delta=2, lr=0.01, iter_max=100, verbose=True)

# Perform optimization
snake.gradient_descent(plot_skip=10)  # plot_skip defines how often to visualize the progress

# Visualize the final snake
snake.plot_snake()

# Plot the evolution of the snake as an animation
snake.plot_evolution()
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