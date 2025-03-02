import os  
import time  
import cv2  
import albumentations as A  
import numpy as np  
import tensorflow as tf  
import matplotlib.pyplot as plt  

class TennisBallDetector:  
    def __init__(self, input_shape=(480, 620, 3), num_classes=2):  
        self.input_shape = input_shape  
        self.num_classes = num_classes  
        self.model = None  

        # Define augmentation transform  
        self.augmentation = A.Compose([  
            A.NoOp(p=1.0),  
            A.RandomBrightnessContrast(p=0.5),  
            A.HueSaturationValue(  
                hue_shift_limit=15,   
                sat_shift_limit=15,  
                val_shift_limit=30,  
                p=0.5  
            ),  
            A.RandomFog(p=0.5),  
            A.RandomShadow(p=0.5),  
            A.GaussNoise(p=0.5),  
            A.Blur(blur_limit=3, p=0.5),  
            A.RandomBrightnessContrast(  
                brightness_limit=(-1, -0.7),  
                contrast_limit=(-1, -0.7),    
                p=1.0  
            ),  
        ], bbox_params=A.BboxParams(format='yolo', label_fields=[]))  

        self.falseAugmentation = A.Compose([  
            A.ColorJitter(  
                brightness=0.8,  
                contrast=0.8,  
                saturation=0.8,  
                hue=0.8,  
                p=1.0  
            ),  
            A.HueSaturationValue(  
                hue_shift_limit=(-30, 30),  
                sat_shift_limit=(-50, -20),   
                val_shift_limit=(-30, 30),  
                p=1.0  
            ),  
        ], bbox_params=A.BboxParams(format='yolo', label_fields=[]))  

    def load_and_preprocess_data(self, image_dir, annotation_dir):  
        """  
        Load images and annotations from directories.  

        Expected structure:  
        - image_dir: contains all image files  
        - annotation_dir: contains corresponding annotation files  
        """  
        images = []          
        coordinates = []  

        # List all image files  
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]  

        for image_file in image_files:  
            # Full paths  
            image_path = os.path.join(image_dir, image_file)  
            
            # Find corresponding annotation file  
            annotation_file = os.path.join(annotation_dir, os.path.splitext(image_file)[0] + '.txt')  
            
            # Read and preprocess image  
            try:  
                # Read image  
                img = cv2.imread(image_path)  
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                
                # Read annotation  
                if os.path.exists(annotation_file):  
                    with open(annotation_file, 'r') as f:  
                        annotations = f.readlines()  
                    
                    for line in annotations:  
                        # Parse YOLO format annotation  
                        parts = line.strip().split()  
                        
                        height, width, _ = img.shape  

                        # Extract annotation details  
                        class_id = int(float(parts[0]))  
                        ymin = min(max(float(parts[1]), 0), 1)  * height  
                        xmin = min(max(float(parts[2]), 0), 1)  * width  
                        ymax = min(max(float(parts[3]), 0), 1)  * height  
                        xmax = min(max(float(parts[4]), 0), 1)  * width  
                        
                        # Compute center, width, height                          
                        x_center = (xmin + xmax) / (2 * width)  
                        y_center = (ymin + ymax) / (2 * height)  
                        box_width = (xmax - xmin) / width  
                        box_height = (ymax - ymin) / height  

                        # Resize image  
                        resized_img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))  
                        
                        # Store data  
                        images.append(resized_img)                          
                                                
                        # Store normalized coordinates  
                        coordinates.append([x_center, y_center, box_width, box_height])  
            
            except Exception as e:  
                print(f"Error processing {image_file}: {e}")  
        
        return (np.array(images), np.array(coordinates))   

    def augment_data(self, images, coordinates, num_augmentations=3):  
        """  
        Augment images and their corresponding coordinates.  

        Args:  
            images: numpy array of original images  
            coordinates: numpy array of normalized coords [x_center, y_center, width, height]  
            num_augmentations: number of augmentations per image  

        Returns:  
            augmented_images: numpy array of augmented images  
            augmented_coordinates: numpy array of augmented coordinates  
            augmented_validity: one-hot labels [Ball, No Ball]  
        """  
        augmented_images = []  
        augmented_coordinates = []  
        augmented_validity = []  

        for validity in [True, False]:  
            for img, coord in zip(images, coordinates):  
                for _ in range(num_augmentations):  
                    # Prepare bounding box for augmentation  
                    x_center, y_center, box_width, box_height = coord  

                    if validity:  
                        augmented = self.falseAugmentation(  
                            image=img,  
                            bboxes=[[x_center, y_center, box_width, box_height]],  
                        )  
                    else:  
                        augmented = self.augmentation(  
                            image=img,  
                            bboxes=[[x_center, y_center, box_width, box_height]],  
                        )  

                    aug_img = augmented['image']  
                    augmented_images.append(aug_img)  
                    augmented_coordinates.append(coord)  
                    label = [1, 0] if validity else [0, 1]  
                    augmented_validity.append(label)  

        return (np.array(augmented_images),  
                np.array(augmented_coordinates),  
                np.array(augmented_validity))  

    def split_data(self, images, labels, coordinates, train_split=0.9):  
        """Split data into train and validation sets."""  
        split_idx = int(len(images) * train_split)  
        X_train = images[:split_idx]  
        y_train_class = labels[:split_idx]  
        y_train_coord = coordinates[:split_idx]  

        X_val = images[split_idx:]  
        y_val_class = labels[split_idx:]  
        y_val_coord = coordinates[split_idx:]  

        return (  
            X_train,  
            {'class_output': y_train_class, 'coordinate_output': y_train_coord},  
            X_val,  
            {'class_output': y_val_class, 'coordinate_output': y_val_coord},  
        )  

    def create_model(self, train_images_count):  
        """Create the MobileNetV2-based model with two outputs:  
        - class_output: for ball vs. no-ball classification  
        - coordinate_output: for bounding box coordinates  
        """  
        input_shape = (224, 224, 3)  

        base_model = tf.keras.applications.MobileNetV2(  
            weights='imagenet',  
            include_top=False,  
            input_shape=input_shape  
        )  
        base_model.trainable = False  

        inputs = tf.keras.Input(shape=input_shape)  
        x = base_model(inputs)  
        x = tf.keras.layers.GlobalAveragePooling2D()(x)  

        x = tf.keras.layers.Dense(256, activation='relu')(x)  
        x = tf.keras.layers.Dropout(0.5)(x)  
        x = tf.keras.layers.Dense(128, activation='relu')(x)  
        x = tf.keras.layers.Dropout(0.3)(x)  

        classification_output = tf.keras.layers.Dense(  
            self.num_classes,  
            activation='softmax',  
            name='class_output'  
        )(x)  

        coordinate_output = tf.keras.layers.Dense(  
            4,  
            activation='linear',  
            name='coordinate_output'  
        )(x)  

        model = tf.keras.Model(  
            inputs=inputs,  
            outputs=[classification_output, coordinate_output]  
        )  

        model.compile(  
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  
            loss={  
                'class_output': 'categorical_crossentropy',  
                'coordinate_output': 'mse'  
            },  
            metrics={  
                'class_output': 'accuracy',  
                'coordinate_output': 'mse'  
            }  
        )  
        return model  

    def train(self, X_train, y_train, X_val, y_val):  
        """Train the model using early stopping."""  
        early_stopping = tf.keras.callbacks.EarlyStopping(  
            monitor='val_loss',  
            patience=10,  
            min_delta=0.001,  
            restore_best_weights=True  
        )  

        history = self.model.fit(  
            X_train,  
            {  
                'class_output': y_train['class_output'],  
                'coordinate_output': y_train['coordinate_output']  
            },  
            validation_data=(  
                X_val,  
                {  
                    'class_output': y_val['class_output'],  
                    'coordinate_output': y_val['coordinate_output']  
                }  
            ),  
            epochs=50,  
            batch_size=32,  
            callbacks=[early_stopping]  
        )  
        return history  

    def inference_and_visualize(self, X_val, y_val):  
        """Run inference on validation data and visualize results."""  
        start_time = time.time()  
        predictions = self.model.predict(X_val)  
        end_time = time.time()  
        print(f"Execution time: {end_time - start_time} seconds")  

        for i in range(min(30, len(X_val))):  
            plt.figure(figsize=(10, 10))  # Each image shown in a new plot  

            img = X_val[i]  
            pred_class = np.argmax(predictions[0][i])  
            pred_coords = predictions[1][i]  
            true_class = np.argmax(y_val['class_output'][i])  
            true_coords = y_val['coordinate_output'][i]  

            # Convert to OpenCV image  
            img_cv = (img * 255).astype(np.uint8)  

            # Draw true coordinates (blue) if labeled as ball  
            if true_class == 0:  
                x_true, y_true, w_true, h_true = true_coords  
                x_true = int(x_true * img_cv.shape[1])  
                y_true = int(y_true * img_cv.shape[0])  
                radius_true = int(max(w_true * img_cv.shape[1], h_true * img_cv.shape[0]) / 2)  
                cv2.circle(img_cv, (x_true, y_true), radius_true, (255, 0, 0), 2)  # Blue circle  

            # Draw predicted coordinates (red) if classified as ball  
            if pred_class == 0:  
                x, y, w, h = pred_coords  
                x = int(x * img_cv.shape[1])  
                y = int(y * img_cv.shape[0])  
                radius = int(max(w, h) * min(img.shape[:2]) / 2)  
                cv2.circle(img_cv, (x, y), radius, (0, 0, 255), 2)  # Red circle  

            plt.imshow(img_cv)  
            plt.title(f'Pred: {"Ball" if pred_class == 0 else "No Ball"}\n'  
                      f'True: {"Ball" if true_class == 0 else "No Ball"}')  
            plt.axis('off')  

        plt.tight_layout()  
        plt.show()  

def main():  
    IMAGE_DIR = '/mnt/d/work/self_drive/tracking/data_set_2/images'  
    ANNOTATION_DIR = '/mnt/d/work/self_drive/tracking/data_set_2/annotations'  

    detector = TennisBallDetector()  

    # Load data  
    images, coordinates = detector.load_and_preprocess_data(IMAGE_DIR, ANNOTATION_DIR)  
    augmented_images, augmented_coordinates, augmented_validity = detector.augment_data(images, coordinates)  

    # Shuffle together  
    permutation = np.random.permutation(len(augmented_images))  
    augmented_images = augmented_images[permutation]  
    augmented_coordinates = augmented_coordinates[permutation]  
    augmented_validity = augmented_validity[permutation]  

    # Split for training  
    X_train, y_train, X_val, y_val = detector.split_data(augmented_images, augmented_validity, augmented_coordinates)  

    # Create and train model  
    detector.model = detector.create_model()  
    detector.train(X_train, y_train, X_val, y_val)  

    # Use the validation set for inference/visualization  
    detector.inference_and_visualize(X_val, y_val)  

if __name__ == '__main__':  
    main()