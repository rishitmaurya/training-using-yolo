import os
import sys
import shutil
from labelme2yolo import Labelme2YOLO  # Assuming the original code is saved as labelme2yolo.py

def convert_labelme_to_yolo(json_dir, output_dir, val_size=0.1, to_seg=False):
    # Create an instance of the Labelme2YOLO class
    convertor = Labelme2YOLO(json_dir, to_seg=to_seg)
    
    # Make sure the output directories exist
    labelled_txt_dir = os.path.join(output_dir, 'labelled_txt')
    if not os.path.exists(labelled_txt_dir):
        os.makedirs(labelled_txt_dir)

    # Run the conversion for all JSON files in the directory
    convertor.convert(val_size=val_size)

    # Move the YOLO label files (txt) to the 'labelled_txt' folder
    label_dir_path = os.path.join(convertor._save_path_pfx, 'labels')
    for folder in ['train', 'val']:
        yolo_label_dir = os.path.join(label_dir_path, folder)
        if os.path.exists(yolo_label_dir):
            for file_name in os.listdir(yolo_label_dir):
                if file_name.endswith('.txt'):
                    source_path = os.path.join(yolo_label_dir, file_name)
                    destination_path = os.path.join(labelled_txt_dir, file_name)
                    shutil.move(source_path, destination_path)
    
    print(f"Conversion complete. YOLO labels are saved in {labelled_txt_dir}")

if __name__ == '__main__':
    # Define paths for your input and output directories
    json_dir = 'datasets_new/labels'  # Replace with your actual path
    output_dir = 'datasets_new/labels_new'  # Replace with your desired output path

    # Call the function to convert LabelMe annotations to YOLO format
    convert_labelme_to_yolo(json_dir, output_dir, val_size=0.1, to_seg=False)
