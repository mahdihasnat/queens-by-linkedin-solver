import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_grid_colors_with_clustering(image_path, grid_size, num_colors=8):
    # Load the image
    image = cv2.imread(image_path)


    # Convert the image to grayscale for contour detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # visualize grayscale image
    cv2.imshow('gray', gray)

    # Apply Gaussian blur to reduce noise (optional)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # visualize blurred image
    cv2.imshow('blurred', blurred)

    # Apply adaptive threshold to get a binary image
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

    # visualize threshold image
    cv2.imshow('thresh', thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imshow('contours', image)

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

    # draw the largest square contour
    cv2.drawContours(image, [largest_square_contour], -1, (0, 0, 255), 3)
    cv2.imshow('largest_square_contour', image)
    cv2.waitKey(0)

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

# Extract the grid colors with clustering
colors = extract_grid_colors_with_clustering(image_path)

# Print the resulting 2D array of colors
if colors is not None:
    for row in colors:
        print(row)
