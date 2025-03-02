import os  
import numpy as np  
import tensorflow as tf  
import random  
import cv2  
import matplotlib.pyplot as plt  
import albumentations as A  

class InferenceHandler:  
    #def __init__(self, model_path, images_dir, annotations_dir):  
    def __init__(self, model, images_dir, annotations_dir):  
        # Load the saved model  
        #self.model = tf.keras.models.load_model(model_path)  
        self.model = model
        self.images_dir = images_dir  
        self.annotations_dir = annotations_dir  
        self.augmentations = self.create_augmentations()  

    def create_augmentations(self):  
        """Create a variety of image augmentations."""  
        return A.Compose([  
            A.RandomRotate90(p=0.5),  
            A.HorizontalFlip(p=0.5),  
            A.RandomBrightnessContrast(p=0.3),  
            A.Blur(blur_limit=3, p=0.2),  
            A.RandomCrop(width=250, height=250, p=0.3),  
            A.Perspective(scale=(0.05, 0.1), p=0.2),  
            A.GaussNoise(p=0.2)  
        ])  

    def load_and_preprocess_image(self, image_path):  
        """Load and preprocess image for model input."""  
        # Read original image  
        original_image = cv2.imread(image_path)  
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  
        
        # Resize and normalize  
        resized_image = cv2.resize(original_image, (300, 300))  
        normalized_image = resized_image / 255.0  
        
        return original_image, normalized_image  

    def apply_augmentation(self, image):  
        """Apply random augmentation to the image."""  
        augmented = self.augmentations(image=image)  
        return augmented['image']  

    def predict_and_visualize(self, num_images=20):  
        """  
        Select random images, augment them, and visualize predictions.  
        """  
        # Get list of image files  
        image_files = [f for f in os.listdir(self.images_dir)   
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  
        
        # Randomly select images  
        selected_images = random.sample(image_files, min(num_images, len(image_files)))  
        
        # Setup plot  
        plt.figure(figsize=(20, 15))  
        
        for i, filename in enumerate(selected_images):  
            # Full image path  
            image_path = os.path.join(self.images_dir, filename)  
            
            # Load original and preprocessed image  
            original_image, normalized_image = self.load_and_preprocess_image(image_path)  
            
            # Apply augmentation  
            augmented_image = self.apply_augmentation(original_image)  
            augmented_normalized = cv2.resize(augmented_image, (300, 300)) / 255.0  
            
            # Predict (now with multiple outputs)  
            bbox_pred, class_pred = self.model.predict(np.expand_dims(augmented_normalized, axis=0))  
            
            # Subplot  
            plt.subplot(5, 10, i+1)  
            
            # Draw the augmented image  
            plt.imshow(augmented_image)  
            
            # Get prediction confidence and class  
            confidence = np.max(class_pred[0])  
            predicted_class_id = np.argmax(class_pred[0])  
            
            # Unpack predicted bounding box  
            # Assuming bbox_pred is in [ymin, xmin, ymax, xmax] format  
            ymin, xmin, ymax, xmax = bbox_pred[0]  
            print ("y min %f, x min %f, ymax %f, xmax %f" % (ymin, xmin, ymax, xmax))

            # Convert to pixel coordinates  
            h, w = augmented_image.shape[:2]  
            left = int(xmin * w)  
            top = int(ymin * h)  
            right = int(xmax * w)  
            bottom = int(ymax * h)  
            
            # Draw predicted bounding box in red  
            cv2.rectangle(augmented_image,   
                        (left, top),   
                        (right, bottom),   
                        (255, 0, 0), 2)  # Red color  
            
            # Try to load ground truth bounding box for comparison  
            try:  
                # Attempt to load annotation  
                annotation_filename = os.path.splitext(filename)[0] + '.txt'  
                annotation_path = os.path.join(self.annotations_dir, annotation_filename)  
                
                with open(annotation_path, 'r') as f:  
                    # Read first line (assuming single object per image)  
                    parts = f.readline().strip().split()  
                    ground_truth_class_id = int(float(parts[0]))  
                    
                    # Ground truth coordinates  
                    gt_ymin = float(parts[1])  
                    gt_xmin = float(parts[2])  
                    gt_ymax = float(parts[3])  
                    gt_xmax = float(parts[4])  
                    
                    # Convert ground truth to pixel coordinates  
                    gt_left = int(gt_xmin * w)  
                    gt_top = int(gt_ymin * h)  
                    gt_right = int(gt_xmax * w)  
                    gt_bottom = int(gt_ymax * h)  
                    
                    # Draw ground truth box in green  
                    cv2.rectangle(augmented_image,   
                                (gt_left, gt_top),   
                                (gt_right, gt_bottom),   
                                (0, 255, 0), 2)  # Green color  
            except Exception as e:  
                print(f"Could not load annotation for {filename}: {e}")  
            
            # Draw prediction confidence and class  
            plt.title(  
                f'Pred: {predicted_class_id}\nConf: {confidence:.2f}',   
                fontsize=8  
            )  
            plt.axis('off')  
        
        plt.tight_layout()  
        plt.show() 


class Trainer:  
    def __init__(self, images_dir, annotations_dir, num_classes, epochs=50):  
        self.images_dir = images_dir  
        self.annotations_dir = annotations_dir  
        self.num_classes = num_classes  
        self.epochs = epochs  
        self.model = self.prepare_model()  
    
    def prepare_model(self):  
        """Prepare the model for object detection."""  
        input_shape = (300, 300, 3)  
        
        # Define input layer explicitly  
        inputs = tf.keras.Input(shape=input_shape, name='input_layer')  
        
        # Use a pre-trained base model  
        base_model = tf.keras.applications.MobileNetV2(  
            input_shape=input_shape,  
            include_top=False,  
            weights='imagenet'  
        )  
        base_model.trainable = False  
        
        # Connect input to base model  
        x = base_model(inputs)  
        
        # Object detection head  
        x = tf.keras.layers.GlobalAveragePooling2D()(x)  
        x = tf.keras.layers.Dense(512, activation='relu')(x)  
        
        # Outputs for bounding box regression and classification  
        bbox_output = tf.keras.layers.Dense(4, name='bbox_output')(x)  
        class_output = tf.keras.layers.Dense(self.num_classes, activation='sigmoid', name='class_output')(x)  
        
        # Create model with multiple outputs  
        model = tf.keras.Model(  
            inputs=inputs,   
            outputs=[bbox_output, class_output]  
        )  
        
        # Compile with appropriate losses  
        model.compile(  
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  
            loss={  
                'bbox_output': 'mse',  # Mean Squared Error for bounding box regression  
                'class_output': 'categorical_crossentropy'  
            },  
            loss_weights={  
                'bbox_output': 1.0,  
                'class_output': 1.0  
            },  
            metrics={  
                'bbox_output': 'mse',  
                'class_output': 'accuracy'  
            }  
        )  

        return model  

    def load_image(self, image_path):  
        """Load and preprocess image."""  
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(300, 300))  
        normalized_image = tf.keras.preprocessing.image.img_to_array(image) / 255.0  
        return normalized_image  

    def prepare_dataset(self):  
        """Prepare training dataset with images and bounding boxes."""  
        print("Loading images...")  
        images = []  
        bbox_labels = []  # Store bounding box coordinates  
        class_labels = []  # Store class labels  
        
        # Iterate through image files  
        for filename in os.listdir(self.images_dir):  
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
                # Full paths  
                image_path = os.path.join(self.images_dir, filename)  
                annotation_filename = os.path.splitext(filename)[0] + '.txt'  
                annotation_path = os.path.join(self.annotations_dir, annotation_filename)  
                
                # Load image  
                normalized_image = self.load_image(image_path)  
                
                # Read annotation  
                try:  
                    with open(annotation_path, 'r') as f:  
                        for line in f:  
                            parts = line.strip().split()                              
                            class_id = int(float(parts[0]))  

                            # Normalized coordinates (ymin, xmin, ymax, xmax)  
                            ymin = float(parts[1])  
                            xmin = float(parts[2])  
                            ymax = float(parts[3])  
                            xmax = float(parts[4])  
                            
                            # Append image, bounding box, and class  
                            images.append(normalized_image)  
                            
                            # Normalized bounding box coordinates for the model  
                            bbox_labels.append([ymin, xmin, ymax, xmax])  
                            
                            # Ensure class_id is valid  
                            class_id = min(max(class_id, 0), self.num_classes - 1)  
                            
                            # One-hot encode class labels  
                            class_label = tf.keras.utils.to_categorical(  
                                class_id,   
                                num_classes=self.num_classes  
                            )  
                            class_labels.append(class_label)  
                
                except Exception as e:  
                    print(f"Error processing {filename}: {e}")  
        
        # Convert to numpy arrays  
        images = np.array(images)  
        bbox_labels = np.array(bbox_labels)  
        class_labels = np.array(class_labels)  

        # Create TensorFlow Dataset  
        dataset = tf.data.Dataset.from_tensor_slices(  
            (  
                images,   
                {  
                    'bbox_output': bbox_labels,   
                    'class_output': class_labels  
                }  
            )  
        )  
        
        # Shuffle and batch the dataset  
        dataset = (  
            dataset  
            .shuffle(buffer_size=len(images))  
            .batch(32)  
            .prefetch(tf.data.AUTOTUNE)  
        )  

        return dataset  

    def train_model(self, dataset):  
        """Train the model using the prepared dataset."""  
        # Train the model  
        history = self.model.fit(  
            dataset,  
            epochs=self.epochs  
        )  
        
        return history  

# Main function for running the training  
def main():  
    images_dir = '/mnt/d/work/self_drive/tracking/data_set_2/aug_images'  # Update with your image directory  
    annotations_dir = '/mnt/d/work/self_drive/tracking/data_set_2/aug_annotations'  # Update with your annotations directory  
    num_classes = 1  # Update based on your case, all balls belong to one class  
    trainer = Trainer(images_dir, annotations_dir, num_classes)  
    
    # Prepare the full dataset  
    dataset = trainer.prepare_dataset()  
    
    # Train the model  
    history = trainer.train_model(dataset)  
        
    # Create inference handler  
    inference_handler = InferenceHandler(trainer.model, images_dir, annotations_dir)  
    
    # Predict and visualize  
    inference_handler.predict_and_visualize()  

if __name__ == "__main__":  
    main()