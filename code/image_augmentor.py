import os  
import cv2  
import numpy as np  
import albumentations as A  

class TFLiteImageAugmentor:  
    def __init__(self,   
                 input_image_dir,   
                 input_annotation_dir,   
                 output_image_dir,   
                 output_annotation_dir):  
        self.input_image_dir = input_image_dir  
        self.input_annotation_dir = input_annotation_dir  
        self.output_image_dir = output_image_dir  
        self.output_annotation_dir = output_annotation_dir  
        
        # Create output directories  
        os.makedirs(output_image_dir, exist_ok=True)  
        os.makedirs(output_annotation_dir, exist_ok=True)  
        
        # Augmentation pipeline  
        self.transform = A.Compose([  
            # Always apply no-op transform 
            A.NoOp(p=1.0)  ,
            # Geometric transformations  
            A.RandomCrop(width=416, height=416, p=1.0),  
            A.HorizontalFlip(p=0.5),  
            A.Rotate(limit=30, p=0.5),  
            
            # Color transformations  
            A.RandomBrightnessContrast(p=0.5),  
            A.HueSaturationValue(p=0.5),  
            
            # Noise and light transformations  
            A.RandomFog(p=0.3),  
            A.RandomShadow(p=0.3),  
            A.GaussNoise(p=0.3)  
        ], bbox_params=A.BboxParams(  
            format='pascal_voc',  # Changed to pascal_voc  
            min_area=0.01,  
            min_visibility=0.1,  
            label_fields=['class_labels']  
        ))  

    def load_tflite_annotations(self, annotation_path, image_width, image_height):  
        """  
        Load annotations and convert to Pascal VOC pixel coordinates  
        """  
        annotations = []  
        with open(annotation_path, 'r') as f:  
            for line in f:  
                # Split the line and convert to float  
                parts = line.strip().split()  
                class_id = int(parts[0])  
                
                # Convert normalized coordinates to pixel coordinates  
                ymin = min(max(float(parts[1]), 0),1) * image_height  
                xmin = min(max(float(parts[2]), 0),1) * image_width  
                ymax = min(max(float(parts[3]), 0),1) * image_height  
                xmax = min(max(float(parts[4]), 0),1) * image_width  
                
                annotations.append([  
                    int(xmin),    # x_min (pixel)  
                    int(ymin),    # y_min (pixel)  
                    int(xmax),    # x_max (pixel)  
                    int(ymax),    # y_max (pixel)  
                    class_id      # class_id  
                ])  
        return annotations  

    def convert_to_normalized(self, bboxes, image_width, image_height):  
        """  
        Convert pixel coordinates back to normalized coordinates  
        """  
        normalized_bboxes = []  
        for bbox in bboxes:  
            x_min, y_min, x_max, y_max, class_id = bbox  
            
            # Convert to normalized coordinates  
            norm_xmin = x_min / image_width  
            norm_ymin = y_min / image_height  
            norm_xmax = x_max / image_width  
            norm_ymax = y_max / image_height  
            
            normalized_bboxes.append([  
                norm_xmin,  
                norm_ymin,  
                norm_xmax,  
                norm_ymax,  
                class_id  
            ])  
        return normalized_bboxes  

    def augment_dataset(self, num_augmentations=5):  
        """  
        Augment entire dataset  
        """  
        for filename in os.listdir(self.input_image_dir):  
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
                # Paths  
                image_path = os.path.join(self.input_image_dir, filename)  
                annotation_filename = os.path.splitext(filename)[0] + '.txt'  
                annotation_path = os.path.join(self.input_annotation_dir, annotation_filename)  
                
                # Read image  
                image = cv2.imread(image_path)  
                image_height, image_width = image.shape[:2]  
                
                # Load annotations  
                try:  
                    # Load annotations in pixel coordinates  
                    annotations = self.load_tflite_annotations(annotation_path, image_width, image_height)  
                except Exception as e:  
                    print(f"Error processing {filename}: {e}")  
                    continue  
                
                # Separate bboxes and labels  
                bboxes = [ann[:4] for ann in annotations]  
                labels = [ann[4] for ann in annotations]  
                
                # Generate augmentations  
                for i in range(num_augmentations):  
                    # Apply augmentation  
                    augmented = self.transform(  
                        image=image,   
                        bboxes=bboxes,   
                        class_labels=labels  
                    )  
                    
                    # New filenames  
                    new_image_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"  
                    new_annotation_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.txt"  
                    
                    # Save augmented image  
                    output_image_path = os.path.join(  
                        self.output_image_dir,   
                        new_image_filename  
                    )  
                    cv2.imwrite(output_image_path, augmented['image'])  
                    
                    # Convert augmented bboxes back to normalized coordinates  
                    augmented_bboxes = self.convert_to_normalized(  
                        # Combine bboxes with their labels  
                        [bbox + [label] for bbox, label in zip(augmented['bboxes'], augmented['class_labels'])],   
                        image_width,   
                        image_height  
                    )  
                    
                    # Save augmented annotations  
                    output_annotation_path = os.path.join(  
                        self.output_annotation_dir,   
                        new_annotation_filename  
                    )  
                    
                    # Write annotations  
                    with open(output_annotation_path, 'w') as f:  
                        for ann in augmented_bboxes:  
                            # Format: class_id ymin xmin ymax xmax  
                            f.write(f"{ann[4]} {ann[1]} {ann[0]} {ann[3]} {ann[2]}\n")  


def main():  
    # Usage  
    augmentor = TFLiteImageAugmentor(  
        input_image_dir='/home/usman/tracking/data_set_2/images',  
        input_annotation_dir='/home/usman/tracking/data_set_2/annotations',  
        output_image_dir='/home/usman/tracking/data_set_2/aug_images',  
        output_annotation_dir='/home/usman/tracking/data_set_2/aug_annotations'  
    )  

    augmentor.augment_dataset(num_augmentations=5)

if __name__ == "__main__":  
    main()
