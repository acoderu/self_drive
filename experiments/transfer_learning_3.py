import os  
import numpy as np  
import tensorflow as tf  
import keras.preprocessing.image  
import keras.applications.mobilenet_v2  
import keras.layers  
import keras.models  
from sklearn.model_selection import train_test_split  
import cv2  
import matplotlib.pyplot as plt  
from keras.applications.mobilenet_v2 import MobileNetV2  
from keras.layers import GlobalAveragePooling2D
class TennisBallDetector:  
    def __init__(self, images_dir, annotations_dir):  
        self.images_dir = images_dir  
        self.annotations_dir = annotations_dir  
        self.input_shape = (300, 300, 3)  
        self.model = None  

    def load_data(self):  
        """  
        Load images and their corresponding annotations  
        """  
        images = []  
        bboxes = []  
        labels = []  

        # Get list of image files  
        image_files = [f for f in os.listdir(self.images_dir)   
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]  

        for filename in image_files:  
            # Load image  
            image_path = os.path.join(self.images_dir, filename)  
            image = cv2.imread(image_path)  
            if image is None:  
                print(f"Could not read image: {image_path}")  
                continue  
            image = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))  
            image = image / 255.0  # Normalize  

            # Load corresponding annotation  
            annotation_filename = os.path.splitext(filename)[0] + '.txt'  
            annotation_path = os.path.join(self.annotations_dir, annotation_filename)  

            try:  
                with open(annotation_path, 'r') as f:  
                    # Read first line (assuming single object per image)  
                    parts = f.readline().strip().split()  
                    if len(parts) != 5:  
                        print(f"Invalid annotation format for {annotation_filename}")  
                        continue  
                    # Annotation format: class_id, ymin, xmin, ymax, xmax  
                    class_id = 0 #int(float(parts[0]))  
                    ymin = float(parts[1])  
                    xmin = float(parts[2])  
                    ymax = float(parts[3])  
                    xmax = float(parts[4])  

                    images.append(image)  
                    bboxes.append([ymin, xmin, ymax, xmax])  
                    labels.append(class_id)  

            except Exception as e:  
                print(f"Could not load annotation for {filename}: {e}")  

        return np.array(images), np.array(bboxes), np.array(labels)  

    def prepare_model(self, num_classes):  
        """  
        Prepare transfer learning model based on MobileNetV2  
        """  
        # Base model  
        base_model = MobileNetV2(  
            weights='imagenet',   
            include_top=False,   
            input_shape=self.input_shape  
        )  

        # Freeze base model layers  
        for layer in base_model.layers:  
            layer.trainable = False  

        # Add custom layers  
        x = base_model.output  
        x = GlobalAveragePooling2D()(x)  
        x = tf.keras.layers.Dense(256, activation='relu')(x)  

        # Separate outputs for classification and bounding box  
        class_output = tf.keras.layers.Dense(num_classes, activation='sigmoid', name='class_output')(x)  
        bbox_output = tf.keras.layers.Dense(4, activation='softmax', name='bbox_output')(x)  

        # Create model  
        self.model = tf.keras.models.Model(  
            inputs=base_model.input,   
            outputs=[bbox_output, class_output]  
        )  

        # Compile model  
        self.model.compile(  
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  
            loss={  
                'bbox_output': 'mse',  # Mean Squared Error for bounding box  
                'class_output': 'sparse_categorical_crossentropy'  
            },  
            loss_weights={  
                'bbox_output': 1.0,  
                'class_output': 1.0  
            },  
            metrics={  
                'class_output': 'accuracy'  
            }  
        )  

    def train(self, test_size=0.1, random_state=42):  
        """  
        Train the model with train-test split  
        """  
        # Load data  
        X, y_bbox, y_class = self.load_data()  

        # One-hot encode class labels  
        y_class = tf.keras.utils.to_categorical(y_class)  

        # Split data  
        (X_train, X_test,   
         y_bbox_train, y_bbox_test,   
         y_class_train, y_class_test) = train_test_split(  
            X, y_bbox, y_class,   
            test_size=test_size,   
            random_state=random_state  
        )  

        # Prepare model (assuming binary or multi-class classification)  
        self.prepare_model(num_classes=y_class.shape[1])  

        # Print dataset sizes  
        print(f"Training data size: {X_train.shape[0]}")  
        print(f"Testing data size: {X_test.shape[0]}")  

        # Training  
        history = self.model.fit(  
            X_train,   
            {'bbox_output': y_bbox_train, 'class_output': y_class_train},  
            validation_data=(X_test, {'bbox_output': y_bbox_test, 'class_output': y_class_test}),  
            epochs=50,  
            batch_size=32  
        )  

        return history  

    def predict_and_visualize(self, num_images=5):  
        """  
        Perform inference on test images and visualize results  
        """  
        # Load test data  
        X, y_bbox, y_class = self.load_data()  
        
        # Split test data  
        _, X_test, _, y_bbox_test, _, y_class_test = train_test_split(  
            X, y_bbox, y_class,   
            test_size=0.1  ,
            random_state=42
        )  

        # Predict on test images  
        bbox_pred, class_pred = self.model.predict(X_test)  

        # Visualization  
        plt.figure(figsize=(20, 15))  

        for i in range(min(num_images, len(X_test))):  
            # Prepare subplot  
            plt.subplot(5, 10, i+1)  

            # Denormalize and display image  
            test_image = X_test[i]  
            plt.imshow(test_image)  

            # Predicted bounding box  
            pred_bbox = bbox_pred[i]  
            h, w = test_image.shape[:2]  

            # Convert normalized coordinates to pixel coordinates  
            left = int(pred_bbox[1] * w)  
            top = int(pred_bbox[0] * h)  
            right = int(pred_bbox[3] * w)  
            bottom = int(pred_bbox[2] * h)  

            # Draw predicted bounding box  
            cv2.rectangle(test_image,   
                          (left, top),   
                          (right, bottom),   
                          (255, 0, 0), 2)  

            # Predicted class and confidence  
            predicted_class_id = np.argmax(class_pred[i])  
            confidence = np.max(class_pred[i])  

            plt.title(  
                f'Class: {predicted_class_id}\nConf: {confidence:.2f}',   
                fontsize=8  
            )  
            plt.axis('off')  

        plt.tight_layout()  
        plt.show()  

# Usage example  
if __name__ == '__main__':  
    detector = TennisBallDetector(  
        images_dir='/mnt/d/work/self_drive/tracking/data_set_2/images',   
        annotations_dir='/mnt/d/work/self_drive/tracking/data_set_2/annotations'  
    )  
    
    # Train the model  
    history = detector.train()  
    
    # Visualize predictions  
    detector.predict_and_visualize()