import numpy as np
from PIL import Image

def apply_transformations(input_path, output_path, environment, frame_of_reference):
    # Open the image
    original_image = Image.open(input_path)

    # Convert image to numpy array for transformation
    image_array = np.array(original_image)

    # Define transformation matrices for rotation, reflection, and translation
    rotation_matrix = np.array([[0, -1], [1, 0]])
    reflection_matrix = np.array([[-1, 0], [0, -1]])
    translation_matrix = np.array([[1, 0, 0], [0, 1, 0]])

    # Apply transformations based on the environment
    if environment == "rotate":
        transformation_matrix = rotation_matrix
    elif environment == "reflect":
        transformation_matrix = reflection_matrix
    elif environment == "translate":
        if frame_of_reference == "global":
            transformation_matrix = np.vstack((translation_matrix, np.array([0, 0, 1])))
        elif frame_of_reference == "local":
            transformation_matrix = translation_matrix
        else:
            print("Unknown frame of reference. No transformations applied.")
            return
    else:
        print("Unknown environment. No transformations applied.")
        return

    # Apply the transformation matrix to the image
    transformed_image = np.dot(image_array, transformation_matrix.T)

    # Convert the transformed numpy array back to an image
    transformed_pil_image = Image.fromarray(transformed_image.astype(np.uint8))

    # Save the transformed image
    transformed_pil_image.save(output_path)

if __name__ == "__main__":
    input_image_path = "input.jpg"   # Replace with your input image path
    output_image_path = "output.jpg" # Replace with the desired output image path
    environment = "rotate"  # Change to "reflect" or "translate" based on your choice
    frame_of_reference = "global"  # Change to "local" if desired

    apply_transformations(input_image_path, output_image_path, environment, frame_of_reference)
    print("Transformation applied and saved successfully!")