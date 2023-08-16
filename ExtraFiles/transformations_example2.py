import numpy as np
from PIL import Image

def apply_environment_transformations(input_path, output_path, environment):
    # Open the image
    original_image = Image.open(input_path)

    # Convert image to numpy array for transformation
    image_array = np.array(original_image)

    # Apply transformations based on the environment
    if environment == "rotate":
        transformed_image = np.rot90(image_array, k=1, axes=(0, 1))
    elif environment == "reflect":
        transformed_image = np.fliplr(image_array)
    elif environment == "translate":
        translation_matrix = np.array([[1, 0, 100], [0, 1, 50]])
        transformed_image = np.roll(image_array, shift=(100, 50), axis=(0, 1))
    else:
        print("Unknown environment. No transformations applied.")
        return

    # Convert the transformed numpy array back to an image
    transformed_pil_image = Image.fromarray(transformed_image)

    # Save the transformed image
    transformed_pil_image.save(output_path)

if __name__ == "__main__":
    input_image_path = "input.jpg"   # Replace with your input image path
    output_image_path = "output.jpg" # Replace with the desired output image path
    environment = "rotate"  # Change to "reflect" or "translate" based on your choice

    apply_environment_transformations(input_image_path, output_image_path, environment)
    print("Transformation applied and saved successfully!")