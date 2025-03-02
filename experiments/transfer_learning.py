import os  
import numpy as np  
import tensorflow as tf  
import matplotlib.pyplot as plt  
import cv2  

class BallDetectionTransferLearning:  
    def __init__(self,   
                 original_model_path,   
                 images_dir,   
                 annotations_dir,   
                 output_dir):  
        """  
        Initialize ball detection transfer learning  
        
        Args:  
            original_model_path (str): Path to original TFLite model  
            images_dir (str): Directory with training images  
            annotations_dir (str): Directory with annotation files  
            output_dir (str): Directory to save trained model  
        """  
        self.original_model_path = original_model_path  
        self.images_dir = images_dir  
        self.annotations_dir = annotations_dir  
        self.output_dir = output_dir  
        
        # Create output directory  
        os.makedirs(output_dir, exist_ok=True)  
        
        # Model parameters  
        self.input_shape = (300, 300, 3)  
        self.num_classes = 91  # Ball detection  
        
        # Prepare data and model  
        self.train_images, self.train_labels, self.bboxes = self.prepare_dataset()  
        self.model = self.build_transfer_model()  

    def load_image(self, image_path):  
        """  
        Load and preprocess a single image  
        
        Args:  
            image_path (str): Path to image file  
        
        Returns:  
            tuple: Preprocessed image and original image  
        """  
        # Read original image  
        original_image = cv2.imread(image_path)  
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  
        
        # Get original image dimensions  
        orig_h, orig_w = original_image.shape[:2]  
        
        # Resize and normalize for model input  
        resized_image = cv2.resize(original_image, (self.input_shape[0], self.input_shape[1]))  
        normalized_image = resized_image / 255.0  # Normalize to [0, 1]  
        
        return normalized_image, original_image, orig_h, orig_w  

    def prepare_dataset(self):  
        """  
        Prepare training dataset  
        
        Returns:  
            tuple: Training images, labels, and bounding boxes  
        """  
        print ("loading images")
        images = []  
        labels = []  
        all_bboxes = []  
        
        test = 0
        # Iterate through image files  
        for filename in os.listdir(self.images_dir):  
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
                # Full paths  
                image_path = os.path.join(self.images_dir, filename)  
                annotation_filename = os.path.splitext(filename)[0] + '.txt'  
                annotation_path = os.path.join(self.annotations_dir, annotation_filename)  
                
                # Load image  
                normalized_image, original_image, orig_h, orig_w = self.load_image(image_path)  
                
                # Read annotation  
                try:  
                    with open(annotation_path, 'r') as f:  
                        
                        bboxes_for_image = []  
                        for line in f:  
                            
                            parts = line.strip().split()                              
                            class_id = int(float(parts[0]))  
                            
                            # Normalized coordinates (ymin, xmin, ymax, xmax)  
                            ymin = float(parts[1])  
                            xmin = float(parts[2])  
                            ymax = float(parts[3])  
                            xmax = float(parts[4])  
                            
                            # Convert to pixel coordinates  
                            left = int(xmin * orig_w)  
                            top = int(ymin * orig_h)  
                            right = int(xmax * orig_w)  
                            bottom = int(ymax * orig_h)  
                            
                            # Store bounding box  
                            bboxes_for_image.append([left, top, right, bottom])  
                        
                        test = test + 1
                        # Append data  
                        images.append(normalized_image)  
                        #labels.append(class_id)  # Assuming class ids start from 1  
                        labels.append(class_id)
                        all_bboxes.append(bboxes_for_image)  
                
                except Exception as e:  
                    print(f"Error processing {filename}: {e}")  
        
        print ("number of files is %d " % (test))
        print ("number of labels is %d " % (len(labels)))
        #for label in labels:  
        #    print(label) 

        # Convert to numpy arrays  
        images = np.array(images)  
        labels = tf.keras.utils.to_categorical(labels, num_classes=self.num_classes)  
        print ("images loaded")
        return images, labels, all_bboxes  

    def build_transfer_model(self):  
        """  
        Build transfer learning model  
        
        Returns:  
            tf.keras.Model: Prepared model for transfer learning  
        """  
        # Base model (MobileNetV2)  
        base_model = tf.keras.applications.MobileNetV2(  
            input_shape=self.input_shape,  
            include_top=False,  
            weights='imagenet'  
        )  
        
        # Freeze base model layers  
        base_model.trainable = False  
        
        # Add custom layers  
        model = tf.keras.Sequential([  
            base_model,  
            tf.keras.layers.GlobalAveragePooling2D(),  
            tf.keras.layers.Dense(256, activation='relu'),  
            tf.keras.layers.Dropout(0.5),  
            tf.keras.layers.Dense(self.num_classes, activation='softmax')  
        ])  
        
        # Compile model  
        model.compile(  
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  
            loss='categorical_crossentropy',  
            metrics=['accuracy']  
        )  
        
        return model  

    def train_model(self,   
                    epochs=50,   
                    batch_size=8,   
                    validation_split=0.2):  
        print ("starting training")
        """  
        Train the transfer learning model  
        
        Args:  
            epochs (int): Number of training epochs  
            batch_size (int): Batch size for training  
            validation_split (float): Portion of data for validation  
        
        Returns:  
            History object from model training  
        """  
        # Data augmentation  
        data_augmentation = tf.keras.Sequential([  
            tf.keras.layers.RandomFlip('horizontal'),  
            tf.keras.layers.RandomRotation(0.2),  
            tf.keras.layers.RandomZoom(0.2),  
        ])  
        
        # Augment training data  
        augmented_images = tf.data.Dataset.from_tensor_slices(  
            (self.train_images, self.train_labels)  
        ).shuffle(buffer_size=len(self.train_images))  
        
        augmented_images = augmented_images.map(  
            lambda x, y: (data_augmentation(x, training=True), y)  
        )  
        
        # Train the model  
        history = self.model.fit(  
            augmented_images,  
            epochs=epochs,  
            batch_size=batch_size #,  
            #validation_split=validation_split  
        )  
        print ("finished training")
        return history  

    def visualize_predictions(self, sample_images=None, sample_bboxes=None):  
        print ("visualizing predictions")
        """  
        Visualize model predictions with original bounding boxes  
        
        Args:  
            sample_images (np.array, optional): Sample images to visualize  
            sample_bboxes (list, optional): Corresponding bounding boxes  
        """  
        if sample_images is None:  
            sample_images = self.train_images[:5]  
            sample_bboxes = self.bboxes[:5]  
        
        plt.figure(figsize=(15, 3))  
        for i, (image, bboxes) in enumerate(zip(sample_images, sample_bboxes)):  
            # Denormalize image  
            display_image = (image * 255).astype(np.uint8)  
            
            # Create subplot  
            plt.subplot(1, len(sample_images), i+1)  
            plt.imshow(display_image)  
            
            # Draw bounding boxes  
            for bbox in bboxes:  
                left, top, right, bottom = bbox  
                plt.gca().add_patch(plt.Rectangle(  
                    (left, top),   
                    right - left,   
                    bottom - top,   
                    fill=False,   
                    edgecolor='red',   
                    linewidth=2  
                ))  
            
            plt.axis('off')  
        
        plt.tight_layout()  
        plt.savefig(os.path.join(self.output_dir, 'sample_predictions.png'))  
        plt.close()  

    def save_model(self,   
                   saved_model_path=None,   
                   tflite_path=None):  
        """  
        Save the trained model  
        
        Args:  
            saved_model_path (str, optional): Path to save TensorFlow SavedModel  
            tflite_path (str, optional): Path to save TFLite model  
        """  
        # Default paths if not provided  
        if saved_model_path is None:  
            saved_model_path = os.path.join(self.output_dir, 'saved_model')  
        
        if tflite_path is None:  
            tflite_path = os.path.join(self.output_dir, 'ball_detection_model.tflite')  
        
        # Save TensorFlow SavedModel  
        self.model.save(saved_model_path)  
        
        # Convert to TFLite  
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)  
        
        # Optional: Quantization  
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  
        converter.target_spec.supported_types = [tf.float16]  
        
        # Convert and save TFLite model  
        tflite_model = converter.convert()  
        with open(tflite_path, 'wb') as f:  
            f.write(tflite_model)  
        
        print(f"Model saved to {saved_model_path}")  
        print(f"TFLite model saved to {tflite_path}")  

def main():  
    # Use GPU if available  
    physical_devices = tf.config.list_physical_devices('GPU')  
    if physical_devices:  
        tf.config.experimental.set_memory_growth(physical_devices[0], True)  
    
    # Initialize transfer learning  
    trainer = BallDetectionTransferLearning(  
        original_model_path='/mnt/d/work/self_drive/tracking/ssd_mobilenet_v2_coco_quant_postprocess.tflite',  
        images_dir='/mnt/d/work/self_drive/tracking/data_set_2/aug_images',  
        annotations_dir='/mnt/d/work/self_drive/tracking/data_set_2/aug_annotations',  
        output_dir='/mnt/d/work/self_drive/tracking/transformed_model/'  
    )  
    
    # Train the model  
    history = trainer.train_model(  
        epochs=50,  
        batch_size=8  
    )  
    
    # Save the model  
    trainer.save_model()

    # Visualize predictions  
    trainer.visualize_predictions()  
      

if __name__ == "__main__":  
    main()