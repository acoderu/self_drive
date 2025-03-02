import os  
import numpy as np  
import tensorflow as tf  
import cv2  
import albumentations as A  
import matplotlib.pyplot as plt  

class TennisBallDetector:  
    def __init__(self, input_shape=(620, 480, 3), num_classes=2):  
        self.input_shape = input_shape  
        self.num_classes = num_classes  
        self.model = self.create_model()  
        # Define augmentation transform  
        self.augmentation = A.Compose([  
            A.RandomBrightnessContrast(p=0.3),  
            A.HueSaturationValue(  
                hue_shift_limit=20,  # Focus on green/yellow hues  
                sat_shift_limit=30,  
                val_shift_limit=20,  
                p=0.5  
            ),  
            A.RandomRotate90(p=0.3),  
            #A.RandomCrop(  
                #height=int(self.input_shape[1]*0.9),    # per the dimensions in param
                #width=int(self.input_shape[0]*0.9),   
            #    height=int(480 * 0.8),    # Correct: use actual image height  
            #    width=int(620 * 0.8),     # Correct: use actual image width 
            #    p=0.3  
            #),  
            A.Blur(blur_limit=3, p=0.2),  
            A.Affine(  
                scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},  
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  
                p=0.3  
            )  
        ], bbox_params=A.BboxParams(format='yolo', label_fields=[])) 
    
    def load_and_preprocess_data(self, image_dir, annotation_dir):  
        """  
        Load images and annotations from directories  
        
        Expected structure:  
        - image_dir: contains all image files  
        - annotation_dir: contains corresponding annotation files  
        """  
        images = []  
        labels = []  
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
                        
                        print(f"Coordinates: ymin={ymin}, xmin={xmin}, xmax={xmax}, ymax={ymax}")   

                        # Compute center, width, height                          
                        x_center = (xmin + xmax) / (2 * width)  
                        y_center = (ymin + ymax) / (2 * height)  
                        box_width = (xmax - xmin) / width  
                        box_height = (ymax - ymin) / height  

                        print(f"Coordinates: x_center={x_center}, y_center={y_center}, box_width={box_width}, box_height={box_height}")   
                        
                        # Resize image  
                        resized_img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))  
                        #resized_img = resized_img / 255.0  # Normalize  
                        
                        # Store data  
                        images.append(resized_img)  
                        
                        # One-hot encode class (assuming 0 is tennis ball)  
                        label = [1, 0] if class_id == 0 else [0, 1]  
                        labels.append(label)  
                        
                        # Store normalized coordinates  
                        coordinates.append([x_center, y_center, box_width, box_height])  
            
            except Exception as e:  
                print(f"Error processing {image_file}: {e}")  
        
        return (  
            np.array(images),   
            np.array(labels),   
            np.array(coordinates)  
        )  
    
    def create_model(self):  
        # Base model  
        base_model = tf.keras.applications.MobileNetV2(  
            weights='imagenet',   
            include_top=False,  
            input_shape=self.input_shape  
        )  
        base_model.trainable = False  
        
        # Model architecture  
        inputs = tf.keras.Input(shape=self.input_shape)  
        
        # Feature extraction  
        x = base_model(inputs)  
        x = tf.keras.layers.GlobalAveragePooling2D()(x)  
        
        # Hidden layers with dropout  
        x = tf.keras.layers.Dense(256, activation='relu')(x)  
        x = tf.keras.layers.Dropout(0.5)(x)  
        x = tf.keras.layers.Dense(128, activation='relu')(x)  
        x = tf.keras.layers.Dropout(0.3)(x)  
        
        # Multi-output  
        # Classification output  
        classification_output = tf.keras.layers.Dense(  
            self.num_classes,   
            activation='softmax',   
            name='class_output'  
            )(x)  
        
        # Coordinate output  
        coordinate_output = tf.keras.layers.Dense(  
            4,  # [x_center, y_center, width, height]  
            activation='linear',   
            name='coordinate_output'  
        )(x)  
        
        # Create model  
        model = tf.keras.Model(  
            inputs=inputs,   
            outputs=[classification_output, coordinate_output]  
        )  
        
        # Compile model  
        model.compile(  
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  
            loss={  
                'class_output': 'categorical_crossentropy',  
                'coordinate_output': 'mse'  
            },  
            loss_weights={  
                'class_output': 1.0,  
                'coordinate_output': 0.5  
            },  
            metrics={  
                'class_output': 'accuracy',  
                'coordinate_output': 'mse'  
            }  
        )  
        
        return model  
    
    
    def ensure_same_shape(self, images, input_shape=(620, 480, 3)):  
        
        newImages = []  
        for image in images:  # Correct iteration syntax  
            # Check image shape, not size  
            if image.shape != input_shape:  
                resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))  
                newImages.append(resized_image)  
            else:  
                newImages.append(image)  
        return newImages 
    
    def convert_coordinates(self, coord, height, width):  
        # Assuming coord is [x_center, y_center, box_width, box_height]  
        x_center, y_center, box_width, box_height = coord  
        
        # Convert to absolute pixel coordinates  
        x_min = max(0, (x_center - box_width/2) * width)  
        y_min = max(0, (y_center - box_height/2) * height)  
        x_max = min(width, (x_center + box_width/2) * width)  
        y_max = min(height, (y_center + box_height/2) * height)  
        
        # Normalize back to [0, 1] range  
        x_min_norm = x_min / width  
        y_min_norm = y_min / height  
        x_max_norm = x_max / width  
        y_max_norm = y_max / height  
    
        return [x_min_norm, y_min_norm, x_max_norm, y_max_norm]  

    def augment_data(self, images, coordinates, num_augmentations=3):  
        """  
        Augment images and their corresponding coordinates  
        
        Args:  
        - images: numpy array of original images  
        - coordinates: numpy array of normalized coordinates [x_center, y_center, width, height]  
        - num_augmentations: number of augmentations per image  
        
        Returns:  
        - augmented_images: numpy array of augmented images  
        - augmented_coordinates: numpy array of augmented coordinates  
        """  

        for i, image in enumerate(images):  
           print(f"Image {i} shape: {image.shape}")  
           print(f"Height: {image.shape[0]}")  
           print(f"Width: {image.shape[1]}")

        augmented_images = []  
        augmented_coordinates = []  
        
        for img, coord in zip(images, coordinates):  
            # Original image  
            augmented_images.append(img)  
            augmented_coordinates.append(coord)  
            
            # Generate augmentations  
            for _ in range(num_augmentations):  
                # Prepare bounding box for augmentation   
                # Convert normalized coordinates to x_min, y_min, x_max, y_max  
                height, width, _ = img.shape  
                x_center, y_center, box_width, box_height = coord  
                
                x_min , y_min, x_max, y_max = self.convert_coordinates(coord, height, width)
                
                print(f"Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")   
                # Apply augmentation  
                augmented = self.augmentation(  
                    image=img,   
                    bboxes=[[x_center, y_center, box_width, box_height]],  
                    #bboxes=[[x_min, y_min, x_max, y_max]],  
                    class_labels=['tennis_ball']  # Add a dummy label if required  
                )  
                
                # Extract augmented image and bbox  
                aug_img = augmented['image']  
                
                # If bbox exists after augmentation  
                if len(augmented['bboxes']) > 0:  
                    new_bbox = augmented['bboxes'][0]  
                    
                    # Convert back to normalized center, width, height  
                    new_x_center = (new_bbox[0] + new_bbox[2]) / 2  
                    new_y_center = (new_bbox[1] + new_bbox[3]) / 2  
                    new_width = new_bbox[2] - new_bbox[0]  
                    new_height = new_bbox[3] - new_bbox[1]  
                    
                    augmented_images.append(aug_img)  
                    augmented_coordinates.append([  
                        new_x_center,   
                        new_y_center,   
                        new_width,   
                        new_height  
                    ])  
        
        #augmented_images = self.ensure_same_shape(augmented_images)
        return (  
            np.array(augmented_images),   
            np.array(augmented_coordinates)  
        )  
    
    def split_data(self, images, labels, coordinates, train_split=0.95):  
        """  
        Split data into train and validation sets  
        """  
        # Shuffle data  
        indices = np.arange(len(images))  
        np.random.shuffle(indices)  
        
        images = images[indices]  
        labels = labels[indices]  
        coordinates = coordinates[indices]  
        
        # Split point  
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
            {'class_output': y_val_class, 'coordinate_output': y_val_coord}  
        )  
    
    def train(self, X_train, y_train, X_val, y_val):  
        # Early stopping and model checkpoint  
        early_stopping = tf.keras.callbacks.EarlyStopping(  
            monitor='val_loss',   
            patience=10,   
            restore_best_weights=True  
        )  
        
        # Train the model  
        history = self.model.fit(  
            X_train,   
            y_train,  
            validation_data=(X_val, y_val),  
            epochs=50,  
            batch_size=32,  
            callbacks=[early_stopping]  
        )  
        
        return history  
    
    def inference_and_visualize(self, X_val, y_val):  
        # Perform inference  
        predictions = self.model.predict(X_val)  
        
        # Visualize results  
        plt.figure(figsize=(20, 10))  
        for i in range(min(10, len(X_val))):  
            plt.subplot(2, 5, i+1)  
            
            # Original image  
            img = X_val[i]  
            
            # Predicted class and coordinates  
            pred_class = np.argmax(predictions[0][i])  
            pred_coords = predictions[1][i]  
            
            # True class and coordinates  
            true_class = np.argmax(y_val['class_output'][i])  
            true_coords = y_val['coordinate_output'][i]  
            
            # Convert to OpenCV image  
            img_cv = (img * 255).astype(np.uint8)  
            
            # Draw true coordinates in dark blue  
            if true_class == 0:  # True ball detected  
                x_true, y_true, w_true, h_true = true_coords  
                x_true = int(x_true * img.shape[1])  
                y_true = int(y_true * img.shape[0])  
                radius_true = int(max(w_true, h_true) * min(img.shape[:2]) / 2)  
                
                cv2.circle(  
                    img_cv,   
                    (x_true, y_true),   
                    radius_true,   
                    (255, 0, 0),  # Dark blue   
                    2  # Circle thickness  
                )  
            
            # Draw predicted circle if ball detected (in red)  
            if pred_class == 0:  # Predicted ball detected  
                x, y, w, h = pred_coords  
                x, y = int(x * img.shape[1]), int(y * img.shape[0])  
                radius = int(max(w, h) * min(img.shape[:2]) / 2)  
                
                cv2.circle(  
                    img_cv,  
                    (x, y),  
                    radius,  
                    (0, 0, 255),  # Dark red  
                    2  # Circle thickness  
                )  
            
            plt.imshow(img_cv)  
            plt.title(f'Pred: {"Ball" if pred_class == 0 else "No Ball"}\n'  
                    f'True: {"Ball" if true_class == 0 else "No Ball"}')  
            plt.axis('off')  
        
        plt.tight_layout()  
        plt.show()  

def visualize_annotations(images, coordinates, num_images=5):  
    """  
    Visualize images with their bounding boxes  
    
    Args:  
    - images: numpy array of images  
    - coordinates: numpy array of bounding box coordinates (ymin, xmin, ymax, xmax)  
    - num_images: number of images to visualize  
    """  
    plt.figure(figsize=(20, 4))  
    
    for i in range(min(num_images, len(images))):  
        # Get image and coordinates  
        img = images[i]  
        
        # Unpack coordinates (ymin, xmin, ymax, xmax)  
        x_center, y_center, box_width, box_height = coordinates[i]  
        
        # Create a copy of the image for drawing  
        img_draw = img.copy()  
        
        # Debug print  
        print(f"Image {i} coordinates:")  
        print(f"Original: {coordinates[i]}")  
        print(f"Image shape: {img_draw.shape}")  
        
        image_height, image_width, _ = img.shape
        x_center_px = int(x_center * image_width)  
        y_center_px = int(y_center * image_height)  
        
        # Calculate circle radius   
        # You can adjust the scaling factor to make the circle larger or smaller  
        radius_x = int(box_width * image_width / 2)  
        radius_y = int(box_height * image_height / 2)  
        
        # Choose the larger radius for a circular representation  
        radius = max(radius_x, radius_y)  
        
        # Optional: Ensure minimum and maximum radius  
        #radius = max(5, min(radius, 50))  # Between 5 and 50 pixels  
        print(f"Coordinates: xcenter={x_center_px}, ycenter={y_center_px}, radius={radius}")   
        
        # Draw rectangle  
        cv2.circle(  
            img_draw,   
            (x_center_px, y_center_px),  # Center point  
            radius,                       # Radius  
            (0, 255, 0),                  # Color (Green)  
            2                             # Thickness  
            )    
        
        # Subplot  
        plt.subplot(1, num_images, i+1)  
        plt.imshow(img_draw) 
        plt.title(f'Image {i}')  
        plt.axis('off')  
    
    plt.tight_layout()  
    plt.show()   

# For augmentation, create a separate normalization step  
def prepare_for_model(images):  
    """  
    Normalize images for model input after augmentation  
    """  
    normalized_images = images.astype(np.float32) / 255.0  
    return normalized_images  

def main():  
    # Paths to your image and annotation directories  
    IMAGE_DIR = '/mnt/d/work/self_drive/tracking/data_set_2/images'
    ANNOTATION_DIR = '/mnt/d/work/self_drive/tracking/data_set_2/annotations'
    num_augmentations = 3
    # Create detector  
    detector = TennisBallDetector()  
    
    # Load data  
    images, labels, coordinates = detector.load_and_preprocess_data(  
        IMAGE_DIR,   
        ANNOTATION_DIR  
    )  
    
    visualize_annotations(images[:5], coordinates[:5])  

    # Augment data  
    augmented_images, augmented_coordinates = detector.augment_data(  
            images[:2],   
            coordinates[:2]  
        )  

    visualize_annotations(augmented_images[:5], augmented_coordinates[:5])  

    # Corresponding labels (repeat for augmented data)  
    augmented_labels = np.repeat(labels, num_augmentations + 1, axis=0)  
    
    # Split data  
    X_train, y_train, X_val, y_val = detector.split_data(  
        augmented_images,   
        augmented_labels,   
        augmented_coordinates  
    )  
    
    # Train model  
    history = detector.train(X_train, y_train, X_val, y_val)  
    
    # Visualize inference  
    detector.inference_and_visualize(X_val, y_val)  

if __name__ == '__main__':  
    main()