import os
import numpy as np

class PolygonProcessor:
    def __init__(self, input_dir, min_points=10, max_points=15, initial_epsilon=0.01):
        self.input_dir = input_dir
        self.min_points = min_points
        self.max_points = max_points
        self.initial_epsilon = initial_epsilon


    def rdp_algorithm(self, points, epsilon):
        """
        Apply the Ramer-Douglas-Peucker algorithm to reduce the number of points in the polygon.
        :param points: List of (x, y) tuples or numpy array of points.
        :param epsilon: The tolerance (the threshold for simplifying the polygon).
        :return: A simplified list of points.
        """
        def perpendicular_distance(point, line_start, line_end):
            # Calculate the perpendicular distance of the point from the line
            num = abs((line_end[1] - line_start[1]) * point[0] - (line_end[0] - line_start[0]) * point[1] +
                      line_end[0] * line_start[1] - line_end[1] * line_start[0])
            den = np.sqrt((line_end[1] - line_start[1]) ** 2 + (line_end[0] - line_start[0]) ** 2)
            return num / den

        def rdp_recurse(start_idx, end_idx, points, epsilon):
            # Recursively apply RDP to simplify the polygon
            if end_idx - start_idx < 2:
                return [points[start_idx], points[end_idx]]

            # Find the point with the maximum perpendicular distance
            max_distance = 0
            index = start_idx
            for i in range(start_idx + 1, end_idx):
                dist = perpendicular_distance(points[i], points[start_idx], points[end_idx])
                if dist > max_distance:
                    index = i
                    max_distance = dist

            # If the max distance is larger than epsilon, recurse
            if max_distance > epsilon:
                left = rdp_recurse(start_idx, index, points, epsilon)
                right = rdp_recurse(index, end_idx, points, epsilon)
                return left[:-1] + right
            else:
                return [points[start_idx], points[end_idx]]

        # Run the RDP algorithm
        return rdp_recurse(0, len(points) - 1, points, epsilon)

    def adjust_polygon_points(self, polygon):
        """
        Adjust the polygon to have between min_points and max_points using the RDP algorithm.
        :param polygon: List of (x, y) tuples representing the polygon points.
        :return: A simplified polygon with a number of points between min_points and max_points.
        """
        epsilon = self.initial_epsilon
        simplified_polygon = self.rdp_algorithm(polygon, epsilon)

        # If the polygon has too many points, simplify further
        while len(simplified_polygon) > self.max_points:
            epsilon *= 1.5  # Increase epsilon to simplify more
            simplified_polygon = self.rdp_algorithm(polygon, epsilon)

        # If the polygon has too few points, decrease epsilon and re-simplify
        while len(simplified_polygon) < self.min_points:
            epsilon *= 0.7  # Decrease epsilon to allow more points
            simplified_polygon = self.rdp_algorithm(polygon, epsilon)

        return simplified_polygon

    def process_file(self, filename):
        """
        Process a single file by adjusting the polygons in it and saving the result.
        :param filename: The name of the file to process.
        :return: None
        """
        input_file_path = os.path.join(self.input_dir, filename)

        # Read the input file and store each line of the .txt into lines
        with open(input_file_path, 'r') as infile:
            lines = infile.readlines()

        # Process the polygons and write to the output file
        with open(input_file_path, 'w') as outfile:
            for line in lines:
                parts = line.strip().split()
                class_id = parts[0]  # The class ID
                points = list(map(float, parts[1:]))  # The list of coordinates
                polygon = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]  # Pairing x and y coordinates

                # Adjust the polygon to have between min_points and max_points
                adjusted_polygon = self.adjust_polygon_points(polygon)

                # Flatten the adjusted polygon back to a list of coordinates
                adjusted_coordinates = [str(coord) for point in adjusted_polygon for coord in point]

                # Write the adjusted polygon to the output file
                outfile.write(f"{class_id} {' '.join(adjusted_coordinates)}\n")

        print(f"Processed {filename} and saved to {input_file_path}")

    def process_directory(self):
        """
        Process all .txt files in the input directory and subdirectories, saving results to the output directory.
        :return: None
        """
        # Walk through the directory and all its subdirectories
        for dirpath, dirnames, filenames in os.walk(self.input_dir):
            for filename in filenames:
                if filename.endswith('.txt'):
                    self.process_file(os.path.relpath(os.path.join(dirpath, filename), self.input_dir))
