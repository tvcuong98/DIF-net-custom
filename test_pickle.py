import pickle
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the .pickle file (replace 'path_to_your_file.pickle' with your file path)
file_path = '<path>/C2RV-CBCT/data/ToothFairy/processed/projections/P4.pickle'
with open(file_path, 'rb') as f:
    data_dict = pickle.load(f)

# Step 2: Extract the 'projs' array
projs = data_dict['projs']
# Step 3: Visualize a middle slice from the 3D array
# Assuming 'projs' is shaped like (n_projections, height, width)
slice_idx = projs.shape[0] // 2  # Middle projection
plt.imshow(projs[slice_idx, :, :], cmap='gray')
plt.title(f'Projection at index {slice_idx}')
plt.axis('off')  # Hide axes
plt.show()

# Step 4: Save the visualization
output_file = 'projection_visualization.png'
plt.imsave(output_file, projs[slice_idx, :, :], cmap='gray')
print(f"Visualization saved as {output_file}")

# Step 5: Print datatype and data range of 'projs'
print(f"Datatype: {projs.dtype}")
print(f"Data range: {np.min(projs)} to {np.max(projs)}")

# Step 6: Print others
print(f"Projs max: {data_dict['projs_max']}")