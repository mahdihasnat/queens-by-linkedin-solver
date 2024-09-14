import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_grid_colors_with_clustering(image_path, grid_size, num_colors=8):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise (optional)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive threshold to get a binary image
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour that looks like the largest square (the grid)
    largest_square_contour = None
    max_area = 0

    for contour in contours:
        # Approximate the contour to a polygon and check if it has 4 sides
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)

        if len(approx) == 4 and area > max_area:
            largest_square_contour = approx
            max_area = area

    if largest_square_contour is None:
        print("Grid not found!")
        return None

    # Get the bounding box of the grid
    x, y, w, h = cv2.boundingRect(largest_square_contour)

    # Crop the grid from the image
    grid = image[y:y+h, x:x+w]

    # Determine the size of each cell
    cell_w = w // grid_size
    cell_h = h // grid_size

    # Collect all the cell colors to apply clustering
    all_colors = []

    for i in range(grid_size):
        for j in range(grid_size):
            # Get the cell coordinates
            cell = grid[i * cell_h: (i + 1) * cell_h, j * cell_w: (j + 1) * cell_w]

            # Calculate the average color in the cell (BGR)
            avg_color = cell.mean(axis=0).mean(axis=0)
            all_colors.append(avg_color)

    # Convert to NumPy array and apply K-Means clustering
    all_colors = np.array(all_colors)
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(all_colors)

    # Get the cluster centers (quantized colors)
    quantized_colors = kmeans.cluster_centers_.astype(int)

    # Create a 2D array to store the color of each cell
    grid_colors = []

    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            # Get the original average color of the cell
            avg_color = all_colors[i * grid_size + j]

            # Find the nearest quantized color (cluster center)
            cluster_idx = kmeans.predict([avg_color])[0]
            quantized_color = quantized_colors[cluster_idx]

            # Convert BGR to RGB for better visualization
            quantized_color_rgb = quantized_color[::-1]

            # Append the quantized color to the row
            row.append(quantized_color_rgb)

        # Append the row to the grid
        grid_colors.append(row)

    return grid_colors

# Set image path and grid size (you can adjust grid size based on image)
image_path = 'screenshot.png'
grid_size = 10  # Update this dynamically if needed
num_colors = 8  # Number of distinct colors to cluster

# Extract the grid colors with clustering
colors = extract_grid_colors_with_clustering(image_path, grid_size, num_colors)

# Print the resulting 2D array of colors
if colors is not None:
    for row in colors:
        print(row)

import matplotlib.pyplot as plt
import numpy as np

def visualize_grid(grid_colors):
    # Get grid dimensions
    rows = len(grid_colors)
    cols = len(grid_colors[0])

    # Create a plot
    fig, ax = plt.subplots()
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks

    # Set the size of the plot based on the number of cells
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)

    # Iterate over the 2D array and draw each cell as a colored rectangle
    for i in range(rows):
        for j in range(cols):
            color = grid_colors[i][j] / 255.0  # Normalize the color values to [0, 1]
            rect = plt.Rectangle([j, rows - i - 1], 1, 1, facecolor=color)  # Draw each rectangle
            ax.add_patch(rect)

    # Display the grid
    plt.gca().set_aspect('equal', adjustable='box')  # Make sure cells are square
    plt.show()

# Example usage of the function
# Assuming 'colors' is the 2D array of colors obtained earlier
if colors is not None:
    visualize_grid(np.array(colors))
