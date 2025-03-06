import cv2
import os

def resize_images_in_directory(input_dir, output_dir, target_size=640):
    """
    Resizes all images in the input directory to the target size (640 pixels for width or height)
    while maintaining the aspect ratio.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to save resized images.
        target_size (int): Target size for the larger dimension (default 640).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the directory
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        
        # Check if it's an image file
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Read the image
            image = cv2.imread(file_path)

            if image is None:
                print(f"Skipping file: {file_name} (not a valid image)")
                continue

            # Get the original dimensions
            original_height, original_width = image.shape[:2]

            # Calculate the scaling factor to resize while maintaining aspect ratio
            if original_width > original_height:
                scale_factor = target_size / original_width
            else:
                scale_factor = target_size / original_height

            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)

            # Resize the image
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Save the resized image to the output directory
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, resized_image)

            print(f"Resized and saved: {output_path}")

if __name__ == "__main__":
    # Define input and output directories
    input_directory = "datasets_new/images"
    output_directory = "datasets_new/resized_images"

    # Resize all images in the input directory
    resize_images_in_directory(input_directory, output_directory)
