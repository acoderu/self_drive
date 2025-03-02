import os  
import numpy as np  
import tensorflow as tf  
import cv2  
import albumentations as A  
import matplotlib.pyplot as plt  
import keras.backend as K 
from sklearn.model_selection import train_test_split  

class TennisBallDetector:  
    def __init__(self, input_shape=(480, 620, 3), num_classes=2):  
        self.input_shape = input_shape  
        self.num_classes = num_classes  
        self.model = None  

        # Define augmentation transform  
        self.augmentation = A.Compose([  
            A.NoOp(p=1.0)  ,
            A.RandomBrightnessContrast(p=0.5),   #good
            A.HueSaturationValue(  
               hue_shift_limit=15,  # Focus on green/yellow hues  
               sat_shift_limit=15,  
               val_shift_limit=30,  
               p=0.5  
            ),  
            A.RandomFog(p=0.5),  
            A.RandomShadow(p=0.5),  
            A.GaussNoise(p=0.5)  ,            
            A.Blur(blur_limit=3, p=0.5),  
            A.RandomBrightnessContrast(  
               brightness_limit=(-1, -0.7),  # Extreme darkening  
               contrast_limit=(-1, -0.7),    # Extreme contrast reduction  
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
               hue_shift_limit=(-30, 30),  # Shift away from yellow  
               sat_shift_limit=(-50, -20),  # Reduce saturation  
               val_shift_limit=(-30, 30),  # Modify brightness  
               p=1.0  
            ),  
        ], bbox_params=A.BboxParams(format='yolo', label_fields=[])) 
    
    def load_and_preprocess_data(self, image_dir, annotation_dir):  
        """  
        Load images and annotations from directories  
        
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
                        
                        print(f"Coordinates: ymin={ymin}, xmin={xmin}, xmax={xmax}, ymax={ymax}")   

                        # Compute center, width, height                          
                        x_center = (xmin + xmax) / (2 * width)  
                        y_center = (ymin + ymax) / (2 * height)  
                        box_width = (xmax - xmin) / width  
                        box_height = (ymax - ymin) / height  

                        print(f"Coordinates: x_center={x_center}, y_center={y_center}, box_width={box_width}, box_height={box_height}")   
                        
                        # Resize image  
                        resized_img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))  
                        #resized_img = resized_img / 255.0  # Normalize  
                        
                        # Store data  
                        images.append(resized_img)                          
                                                
                        # Store normalized coordinates  
                        coordinates.append([x_center, y_center, box_width, box_height])  
            
            except Exception as e:  
                print(f"Error processing {image_file}: {e}")  
        
        return (  
            np.array(images),               
            np.array(coordinates)  
        )  
    
    def focal_loss(self, y_true, y_pred, alpha=0.25, gamma=2.0):  
        """  
        Compute focal loss for binary classification  
        
        Args:  
            y_true: True labels (0 or 1)  
            y_pred: Predicted probabilities  
            alpha: Weighing factor for positive/negative samples  
            gamma: Focusing parameter  
        """  
        # Ensure y_true and y_pred are compatible  
        y_true = tf.cast(y_true, tf.float32)  
        y_pred = tf.cast(y_pred, tf.float32)  
        
        # Clip predictions to prevent log(0)  
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())  
        
        # Focal loss calculation  
        pt_1 = y_pred * tf.cast(tf.equal(y_true, 1), tf.float32)  
        pt_0 = y_pred * tf.cast(tf.equal(y_true, 0), tf.float32)  
        
        # Compute focal loss  
        focal_loss = -(  
            alpha * tf.reduce_mean(tf.pow(1. - pt_1, gamma) * tf.math.log(pt_1 + K.epsilon())) +  
            (1 - alpha) * tf.reduce_mean(tf.pow(pt_0, gamma) * tf.math.log(1. - pt_0 + K.epsilon()))  
        )  
        
        return focal_loss  

    # Custom combined loss function  
    def improved_coordinate_loss(self, y_true, y_pred):  
        # Combination of MSE and Focal Loss  
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        focal_loss_value = self.focal_loss(y_true, y_pred)  
        return mse_loss * 0.7 + focal_loss_value * 0.3  

    def create_warmup_learning_rate(self, total_images, batch_size=32):  
        initial_learning_rate = 0.0003  
        warmup_epochs = 5  
        total_epochs = 50  
        steps_per_epoch = max(1, total_images // batch_size)  

        warmup_steps = warmup_epochs * steps_per_epoch  
        total_steps = total_epochs * steps_per_epoch  

        # Cosine decay schedule with soft start  
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(  
            initial_learning_rate=initial_learning_rate,  
            decay_steps=total_steps,  
            alpha=0.0  # Minimum learning rate ratio  
        )  

        return lr_schedule  

    # def create_model(self, total_num_images):  
    #     input_shape = (224, 224, 3)   
    #     # Base model  
    #     base_model = tf.keras.applications.MobileNetV2(  
    #         weights='imagenet',   
    #         include_top=False,  
    #         input_shape=input_shape  
    #     )  
    #     base_model.trainable = False  
        
    #     # Model architecture  
    #     inputs = tf.keras.Input(shape=input_shape)  
        
    #     # Feature extraction  
    #     x = base_model(inputs)  
    #     x = tf.keras.layers.GlobalAveragePooling2D()(x)  
        
    #     # Hidden layers with dropout  
    #     x = tf.keras.layers.Dense(256, activation='relu')(x)  
    #     x = tf.keras.layers.Dropout(0.5)(x)  
    #     x = tf.keras.layers.Dense(128, activation='relu')(x)  
    #     x = tf.keras.layers.Dropout(0.3)(x)  
        
    #     #contains ball yes or no  
    #     classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='classification')(x)   
                
    #     coordinates_output =  tf.keras.layers.Dense(4, activation='linear', name='bounding_box')(x)  
        
    #     # Create model  
    #     model = tf.keras.Model(  
    #         inputs=inputs,   
    #         outputs=[classification_output, coordinates_output]  
    #     )  
        
    #     # Calculate appropriate parameters  
    #     batch_size = 32  
        
    #     # Choose learning rate strategy  
    #     lr_schedule = self.create_warmup_learning_rate(  
    #         total_images=total_num_images,   
    #         batch_size=batch_size  
    #     )  

    #     # Configurable optimizer  
    #     optimizer = tf.keras.optimizers.Adam(  
    #         learning_rate=lr_schedule,  
    #         beta_1=0.9,  
    #         beta_2=0.999,  
    #         epsilon=1e-07  
    #     )  
        
        
    #     # Compile model  
    #     model.compile(  
    #         optimizer,  
    #         loss={  
    #             'classification': 'binary_crossentropy',   
    #             'bounding_box': self.improved_coordinate_loss  
    #             #'bounding_box': 'mse'
    #         },  
    #         loss_weights={  
    #             'classification': 1.0,  
    #             'bounding_box': 1.0  
    #         },  
    #         metrics={  
    #             'classification': 'accuracy',  
    #             'bounding_box': 'mse'  
    #         }  
    #     )  
        
    #     self.model = model
    #     return model 

    def create_model(self):  
        input_shape = (224, 224, 3)   
        # Base model  
        base_model = tf.keras.applications.MobileNetV2(  
            weights='imagenet',   
            include_top=False,  
            input_shape=input_shape  
        )  
        base_model.trainable = False  
        
        # Model architecture  
        inputs = tf.keras.Input(shape=input_shape)  
        
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
            name='classification'  
            )(x)  
        
        # Coordinate output  
        coordinate_output = tf.keras.layers.Dense(  
            4,  # [x_center, y_center, width, height]  
            activation='linear',   
            name='bounding_box'  
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
                'classification': 'categorical_crossentropy',  
                'bounding_box': 'mse'  
            },  
            loss_weights={  
                'classification': 1.0,  
                'bounding_box': 0.5  
            },  
            metrics={  
                'classification': 'accuracy',  
                'bounding_box': 'mse'  
            }  
        )  
        
        return model  
        
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

    def augment_data(self, images, coordinates, num_augmentations=3 ):  
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
        augmented_validity = []
        for validity in [True, False]:
            for img, coord in zip(images, coordinates):  
                print ("image original shape is %s" % str(img.shape))
                
                
                # Generate augmentations  
                for _ in range(num_augmentations):  
                    # Prepare bounding box for augmentation   
                    # Convert normalized coordinates to x_min, y_min, x_max, y_max  
                    height, width, _ = img.shape  
                    x_center, y_center, box_width, box_height = coord  
                    
                    x_min , y_min, x_max, y_max = self.convert_coordinates(coord, height, width)
                    
                    print(f"Coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")   
                    
                    if validity:
                        augmented = self.falseAugmentation(  
                            image=img,   
                            bboxes=[[x_center, y_center, box_width, box_height]],  
                            #bboxes=[[x_min, y_min, x_max, y_max]],  
                            #class_labels=['tennis_ball']  # Add a dummy label if required  
                        )
                    else:
                        # Apply augmentation  
                        augmented = self.augmentation(  
                            image=img,   
                            bboxes=[[x_center, y_center, box_width, box_height]],  
                            #bboxes=[[x_min, y_min, x_max, y_max]],  
                            #class_labels=['tennis_ball']  # Add a dummy label if required  
                        )  
                    
                    # Extract augmented image and bbox  
                    aug_img = augmented['image']  
                    augmented_images.append(aug_img)  
                    augmented_coordinates.append(coord) #keep original coordinate, don't alter
                    augmented_validity.append(int(validity))  
                    # augmented_coordinates.append(coord)  
                    # # If bbox exists after augmentation  
                    # if len(augmented['bboxes']) > 0:  
                    #     new_bbox = augmented['bboxes'][0]  
                        
                    #     # Convert back to normalized center, width, height  
                    #     new_x_center = (new_bbox[0] + new_bbox[2]) / 2  
                    #     new_y_center = (new_bbox[1] + new_bbox[3]) / 2  
                    #     new_width = new_bbox[2] - new_bbox[0]  
                    #     new_height = new_bbox[3] - new_bbox[1]  
                        
                    #     augmented_images.append(aug_img)  
                    #     augmented_coordinates.append([  
                    #         new_x_center,   
                    #         new_y_center,   
                    #         new_width,   
                    #         new_height  
                    #     ])  
            
        #augmented_images = self.ensure_same_shape(augmented_images)
        #for image in augmented_images:
        #    print (image.shape)
        return (  
            np.array(augmented_images),   
            np.array(augmented_coordinates)  ,
            np.array(augmented_validity).reshape(-1, 1)   
        )  
    
    # def split_data(self, images, labels, coordinates, train_split=0.95):  
    #     # Existing train-test split  
    #     X_train, X_val, y_train_classification, y_val_classification, \
    #     y_train_coordinates, y_val_coordinates = train_test_split(  
    #         images,   
    #         labels,  # Classification labels  
    #         coordinates,  # Coordinate labels  
    #         test_size=0.1,   
    #         random_state=42  
    #     )  

    #     # New label preparation  
    #     y_train = {  
    #         'classification': y_train_classification,  # Binary presence of ball  
    #         'bounding_box': y_train_coordinates       # Coordinates when ball is present  
    #     }  

    #     y_val = {  
    #         'classification': y_val_classification,  
    #         'bounding_box': y_val_coordinates  
    #     }  

    #     return X_train, X_val, y_train, y_val  
    
    def split_data(self, images, labels, coordinates, train_split=0.9):  
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
            {'classification': y_train_class, 'bounding_box': y_train_coord},  
            X_val,   
            {'classification': y_val_class, 'bounding_box': y_val_coord}  
        )  
    
    def prepare_labels(self, y_train, y_val):  
        """  
        Reshape labels to match model expectations  
        """  
        
        # Reshape classification labels  
        y_train_classification = np.array(y_train['classification']).reshape(-1, 1)  
        y_val_classification = np.array(y_val['classification']).reshape(-1, 1)  
        
        # Ensure bounding box coordinates are 2D  
        y_train_coordinates = np.array(y_train['bounding_box'])  
        y_val_coordinates = np.array(y_val['bounding_box'])  
        
        # Create new dictionaries with reshaped labels  
        y_train_prepared = {  
            'classification': y_train_classification,  
            'bounding_box': y_train_coordinates  
        }  
        
        y_val_prepared = {  
            'classification': y_val_classification,  
            'bounding_box': y_val_coordinates  
        }  
        
        return y_train_prepared, y_val_prepared  

    def train(self, X_train, y_train, X_val, y_val):  
        # Early stopping and model checkpoint  
        early_stopping = tf.keras.callbacks.EarlyStopping(  
            monitor='val_loss',  
            patience=10,  # Wait 10 epochs after no improvement  
            min_delta=0.001,  # Minimal change threshold  
            restore_best_weights=True  
        )  


        # Train the model  
        #history = self.model.fit(  
        #    X_train,   
        #    y_train,  
        #    validation_data=(X_val, y_val),  
        #    epochs=50,  
        #    batch_size=32,  
        #    callbacks=[early_stopping]  
        #)  
        y_train, y_val = self.prepare_labels(y_train, y_val)  
        # Diagnostic prints  
        print("Reshaped Training Classification Labels Shape:", y_train['classification'].shape)  
        print("Reshaped Training Bounding Box Labels Shape:", y_train['bounding_box'].shape)  
        

        y_train_classification = y_train["classification"]
        y_train_coordinates = y_train["bounding_box"]
        print("Training Classification Labels Shape:", y_train_classification.shape)  
        print("Training Bounding Box Labels Shape:", y_train_coordinates.shape)  
        #print("Model Classification Output Shape:", self.model.output[0].shape)  
        #print("Model Bounding Box Output Shape:", self.model.output[1].shape)  
        assert y_train['classification'].shape[0] == y_train['bounding_box'].shape[0], "Mismatch in number of samples"  
        
        # history = self.model.fit(  
        #     X_train,  
        #     {'classification': tf.keras.utils.to_categorical(y_train['classification']), 
        #     'bounding_box': y_train['bounding_box']},  
        #     validation_data=(X_val, {  
        #         'classification': tf.keras.utils.to_categorical(y_val['classification']),   
        #         'bounding_box': y_val['bounding_box']  
        #     }),  
        #     epochs=50,  
        #     batch_size=32  
        # )  
        
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


def resize_images_cv2(images, target_size=(224, 224)):      
    # Resize images  
    resized_images = [  
        cv2.resize(  
            img,   
            target_size,  # Note: cv2.resize uses (width, height)     
        ) for img in images  
    ]      
    return np.array(resized_images)  

# For augmentation, create a separate normalization step  
def prepare_for_model(images):  
    """  
    Normalize images for model input after augmentation  
    """  
    normalized_images = images.astype(np.float32) / 255.0  
    return normalized_images  

def ensure_same_shape(images, classification_labels, bounding_boxes):  
    """  
    Validate consistency of image sizes, labels, and bounding boxes.  
    
    Args:  
        images (list or numpy array): List of input images  
        classification_labels (list or numpy array): Corresponding classification labels  
        bounding_boxes (list or numpy array): Corresponding bounding box coordinates  
    
    Returns:  
        tuple: Validated and potentially trimmed (images, classification_labels, bounding_boxes)  
    """  
    # Comprehensive data validation function  
    def print_diagnostic_info():  
        """  
        Print detailed diagnostic information about the datasets  
        """  
        print("\n--- Data Diagnostic Report ---")  
        print(f"Total Images: {len(images)}")  
        print(f"Total Classification Labels: {len(classification_labels)}")  
        print(f"Total Bounding Boxes: {len(bounding_boxes)}")  
        
        # Image size analysis  
        if len(images) > 0:  
            print("\nImage Size Analysis:")  
            image_shapes = [img.shape for img in images]  
            unique_shapes = set(image_shapes)  
            
            print("Unique Image Shapes:")  
            for shape in unique_shapes:  
                count = image_shapes.count(shape)  
                print(f"  {shape}: {count} images")  
            
            # Detailed shape breakdown  
            if len(unique_shapes) > 1:  
                print("\n WARNING: Multiple image shapes detected!")  
        
        # Label distribution  
        if len(classification_labels) > 0:  
            print("\nLabel Distribution:")  
            import numpy as np  
            unique_labels = np.unique(classification_labels, axis=0)  
            for label in unique_labels:  
                count = np.sum(np.all(classification_labels == label, axis=0))  
                print(f"  Label {label}: {count} instances")  
    
    # Perform comprehensive checks  
    def validate_data_integrity():  
        """  
        Perform thorough data integrity checks  
        """  
        # Check if all inputs are lists or numpy arrays  
        if not all(isinstance(data, (list, np.ndarray)) for data in [images, classification_labels, bounding_boxes]):  
            raise TypeError("All inputs must be lists or numpy arrays")  
        
        # Check for empty datasets  
        if len(images) == 0:  
            raise ValueError("No images found in the dataset")  
        
        # Ensure consistent lengths  
        dataset_lengths = [  
            len(images),   
            len(classification_labels),   
            len(bounding_boxes)  
        ]  
        
        if len(set(dataset_lengths)) > 1:  
            print("\n INCONSISTENT DATASET SIZES:")  
            print(f"Images: {len(images)}")  
            print(f"Classification Labels: {len(classification_labels)}")  
            print(f"Bounding Boxes: {len(bounding_boxes)}")  
            
            # Find the minimum length to trim datasets  
            min_length = min(dataset_lengths)  
            print(f"\n🔧 Trimming datasets to {min_length} samples")  
            
            return (  
                images[:min_length],   
                classification_labels[:min_length],   
                bounding_boxes[:min_length]  
            )  
        
        return images, classification_labels, bounding_boxes  
    
       
    # Execute validation steps  
    try:  
        # Print initial diagnostic information  
        print_diagnostic_info()  
        
        # Validate data integrity and trim if necessary  
        validated_images, validated_labels, validated_boxes = validate_data_integrity()  
        
        # Final verification  
        print("\n Data Validation Complete")  
        print(f"Validated Dataset Size: {len(validated_images)} samples")  
        
        return validated_images, validated_labels, validated_boxes  
    
    except Exception as e:  
        print(f" Data Validation Failed: {e}")  
        raise  

def main():  
    # Paths to your image and annotation directories  
    IMAGE_DIR = '/mnt/d/work/self_drive/tracking/data_set_2/images'
    ANNOTATION_DIR = '/mnt/d/work/self_drive/tracking/data_set_2/annotations'
    num_augmentations = 3
    # Create detector  
    detector = TennisBallDetector()  
    
    # Load data  
    #labels are not used really, all labels are for balls
    images, coordinates = detector.load_and_preprocess_data(  
        IMAGE_DIR,   
        ANNOTATION_DIR  
    )  
    
    #visualize_annotations(images[:5], coordinates[:5])  

    # Augment data  
    augmented_images, augmented_coordinates, augmented_validity = detector.augment_data(  
            images,   
            coordinates            
        )  

    dataSz = len(augmented_images)
    augmented_images = augmented_images[:dataSz-10]
    augmented_coordinates = augmented_coordinates[:dataSz-10]
    augmented_validity = augmented_validity[:dataSz-10]
    
    augmented_images = resize_images_cv2(augmented_images)
    augmented_images = prepare_for_model(augmented_images)

    #visualize_annotations(augmented_images[:10], augmented_coordinates[:10])  

    ensure_same_shape(augmented_images,   augmented_validity,   augmented_coordinates  )
    
    # Split data  
    # X_train, X_val, y_train,  y_val = detector.split_data(  
    #     augmented_images,   
    #     augmented_validity,   
    #     augmented_coordinates  
    # )  
    X_train, y_train, X_val, y_val = detector.split_data(  
        augmented_images,   
        augmented_validity,   
        augmented_coordinates  
    )  
    print(f"X train sz={len(X_train)}, Y train={len(y_train)}")   
    print(f"X test sz={len(X_val)}, Y train={len(y_val)}")   
    
    
    #detector.create_model(len(X_train))
    detector.model = detector.create_model()
    # Train model  
    history = detector.train(X_train, y_train, X_val, y_val)  
    
    # Visualize inference  
    #detector.inference_and_visualize(X_val, y_val)  

if __name__ == '__main__':  
    main()