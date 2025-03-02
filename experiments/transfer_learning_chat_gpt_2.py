import os
import time
import cv2
import albumentations as A
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class TennisBallDetector:
    """
    This class is responsible for detecting tennis balls in images.

    It does three main things:
    1) It loads and augments (modifies) images, along with their bounding box info.
    2) It creates a neural network model using MobileNetV2 as the backbone.
    3) It trains the model and can visualize its predictions.

    --- IMPROVEMENTS FOR SPEED AND CORAL TPU ---
    - We reduce the input image size to 160x160 instead of 224x224, to speed up inference.
    - We lower MobileNetV2's 'alpha' to 0.35, further reducing the model’s size.
    - We include instructions for post-training quantization and TFLite conversion for Coral TPU.
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Constructor for the TennisBallDetector class.

        Args:
            input_shape: The desired shape of input images (height, width, channels).
                         Reduced from (224, 224, 3) to (160, 160, 3) for faster inference.
            num_classes: How many classes we want to classify. Here, 2 = [Ball, No Ball].
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None  # Will be set later when we build our model.

        # This is an "augmentation pipeline" for images that DO have tennis balls.
        # Augmentation means randomly changing images (by adjusting brightness, adding noise, etc.)
        # to help our model learn to recognize tennis balls in various conditions.
        self.augmentation = A.Compose([
            #A.NoOp(p=1.0),  # Does nothing; included as a placeholder/option.           
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=10,
                val_shift_limit=10,
                p=0.2
            ),
            A.RandomFog(  
                fog_coef_lower=0.1,    # Minimal fog  
                fog_coef_upper=0.2,    # Light haze  
                alpha_coef=0.05,       # Subtle spread  
                p=0.2                  # 30% chance  
            ) ,
            A.RandomShadow(  
                shadow_roi=(0, 0.5, 1, 1),  # Lower half of image  
                num_shadows_lower=1,         # 1 shadow  
                num_shadows_upper=1,         # Always 1 shadow  
                shadow_dimension=3 ,         # Small shadow  
                p=0.2                        # 30% chance  
            ) ,
            A.GaussNoise(var_limit=(5, 10),  p=0.2),
            A.Blur(blur_limit=2, p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),  # Extreme darkening
                contrast_limit=(-0.2, 0.2),    # Extreme contrast reduction
                p=0.2
            ),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=[]))

        # This is another augmentation pipeline for images that do NOT have tennis balls
        # (or "false" images). Adjusting color, brightness, etc., tries to ensure
        # the model doesn't incorrectly label these as tennis balls.
        self.falseAugmentation = A.Compose([
            A.ColorJitter(
                brightness=0.8,  
                contrast=0.8,  
                saturation=0.8,  
                hue=0.8,  
                p=0.5  
            ),
            A.HueSaturationValue(
                hue_shift_limit=(-50, 50),
                sat_shift_limit=(-50, 50),
                val_shift_limit=(-50, 50),
                p=0.5
            ),
             A.RandomRain(  
                slant_lower=-20,  
                slant_upper=20,  
                drop_length=20,  
                drop_width=2,  
                blur_value=3,  
                brightness_coefficient=0.5,  
                p=0.5  
            )  
        ], bbox_params=A.BboxParams(format='yolo', label_fields=[]))

    def load_and_preprocess_data(self, image_dir, annotation_dir):
        """
        Reads images and their bounding box annotations from folders on disk.

        - image_dir: folder containing .jpg, .png, or .jpeg images.
        - annotation_dir: folder containing .txt files, each with bounding boxes.
          The text files use YOLO format: class_id, ymin, xmin, ymax, xmax (all normalized).
        
        Returns:
            images: A NumPy array of the resized images.
            coordinates: A NumPy array of bounding boxes in normalized format:
                         [x_center, y_center, box_width, box_height].
        """
        images = []
        coordinates = []

        # Get a list of all image files in image_dir.
        image_files = [
            f for f in os.listdir(image_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            annotation_file = os.path.join(
                annotation_dir, os.path.splitext(image_file)[0] + '.txt'
            )

            try:
                # Use OpenCV to read and convert the image from BGR to RGB.
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # If we have a matching annotation file, read the bounding box info.
                if os.path.exists(annotation_file):
                    with open(annotation_file, 'r') as f:
                        annotations = f.readlines()

                    for line in annotations:
                        parts = line.strip().split()

                        # Get the dimensions of the image.
                        height, width, _ = img.shape
                        # YOLO format: parts[0] is class, parts[1..4] are bounding box values in [0..1].
                        class_id = int(float(parts[0]))
                        ymin = min(max(float(parts[1]), 0), 1) * height
                        xmin = min(max(float(parts[2]), 0), 1) * width
                        ymax = min(max(float(parts[3]), 0), 1) * height
                        xmax = min(max(float(parts[4]), 0), 1) * width

                        # Compute x_center, y_center, width, height in normalized form.
                        x_center = (xmin + xmax) / (2 * width)
                        y_center = (ymin + ymax) / (2 * height)
                        box_width = (xmax - xmin) / width
                        box_height = (ymax - ymin) / height

                        # Resize the image so that all images have the same shape.
                        # We use the shape specified in our constructor, default (160, 160, 3).
                        resized_img = cv2.resize(
                            img, (self.input_shape[1], self.input_shape[0])
                        )

                        images.append(resized_img)
                        coordinates.append([x_center, y_center,
                                            box_width, box_height])

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        return np.array(images), np.array(coordinates)

    def merge_data_set(self, images, coordinates, images_no_ball, coordinates_no_ball):
        merged_images = []
        merged_coordinates = []
        merged_validity = []
        label = [1, 0]
        repeated_labels =  [label for _ in range(len(images))]   
        merged_images.extend(images)
        merged_coordinates.extend(coordinates)
        merged_validity.extend(repeated_labels)
        label = [0, 1]
        repeated_labels =  [label for _ in range(len(images))]   
        merged_images.extend(images_no_ball)
        merged_coordinates.extend(coordinates_no_ball)
        merged_validity.extend(repeated_labels)
        return (np.array(merged_images),
                np.array(merged_coordinates),
                np.array(merged_validity))
    
        
    def augment_data(self, images, coordinates, num_augmentations=3):
        """
        Takes each image-bbox pair and applies random transformations.

        Args:
            images: A list or array of images.
            coordinates: A list or array of bounding boxes
                         in the form [x_center, y_center, box_width, box_height].
            num_augmentations: How many times each image will be augmented.

        Returns:
            - A NumPy array of augmented images.
            - A NumPy array of the same bounding boxes (we don't alter the coords here).
            - A NumPy array indicating if the image is "Ball" or "No Ball" in one-hot form (e.g., [1,0] or [0,1]).
        """
        augmented_images = []
        augmented_coordinates = []
        augmented_validity = []

        # We manually pick two "validities": True (meaning "Ball") and False (meaning "No Ball").
        # The idea is to augment the same images but label them differently (Ball vs No Ball),
        # using different augmentation pipelines for each group.
        for validity in [True, False]:
            for img, coord in zip(images, coordinates):
                for _ in range(num_augmentations):
                    x_center, y_center, box_width, box_height = coord

                    # If "validity" is True, apply the "falseAugmentation" pipeline,
                    # otherwise apply the "augmentation" pipeline.
                    if validity:
                        augmented = self.augmentation(
                            image=img,
                            bboxes=[[x_center, y_center, box_width, box_height]],
                        )
                    else:
                        augmented = self.falseAugmentation(
                            image=img,
                            bboxes=[[x_center, y_center, box_width, box_height]],
                        )

                    # The augmented dictionary contains the transformed image.
                    aug_img = augmented['image']

                    # Add the augmented image and the same bounding box to our lists.
                    augmented_images.append(aug_img)
                    augmented_coordinates.append(coord)

                    # "label" is the classification: [1,0] means "Ball", [0,1] means "No Ball".
                    label = [1, 0] if validity else [0, 1]
                    augmented_validity.append(label)
                    

        # label = [1, 0]
        # repeated_labels =  [label for _ in range(len(images))]   
        #augmented_images.extend(images)
        #augmented_coordinates.extend(coordinates)
        #augmented_validity.extend(repeated_labels)
        #print (f"{len(augmented_validity)} , {len(augmented_coordinates)} , {len(augmented_images)}")
        return (np.array(augmented_images),
                np.array(augmented_coordinates),
                np.array(augmented_validity))

    def split_data(self, images, labels, coordinates, train_split=0.9):
        """
        Splits the data into training and validation sets based on a percentage (train_split).

        Args:
            images: array of image data.
            labels: array of classification labels, e.g. [1,0] or [0,1].
            coordinates: array of bounding box data.
            train_split: the fraction to use for training (e.g. 0.9 means 90% train).

        Returns:
            X_train, y_train, X_val, y_val in the correct format for training.
        """
        # Figure out how many samples go into training vs. validation.
        split_idx = int(len(images) * train_split)

        # Split into training portion
        X_train = images[:split_idx]
        y_train_class = labels[:split_idx]
        y_train_coord = coordinates[:split_idx]

        # Split into validation portion
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
        """
        Builds the TensorFlow model using MobileNetV2 as the "base model".
        
        We add:
        - A GlobalAveragePooling layer to reduce the feature map.
        - Two fully connected (Dense) layers to learn new patterns.
        - Two separate outputs:
            1) classification_output: decides "Ball" vs "No Ball"
            2) coordinate_output: predicts [x_center, y_center, box_width, box_height]

        Args:
            train_images_count: number of images in the training set (not currently used here,
                                but could be used for advanced scheduling of learning rate).
        
        Returns:
            A compiled TensorFlow Keras model.

        --- SPEED IMPROVEMENTS ---
        - We set the alpha=0.35 parameter on MobileNetV2 to make the model smaller
          (fewer parameters), thus faster to run on devices like the Coral or Raspberry Pi.
        - We still freeze the base model to preserve pretrained weights.
        """
        # *** Key changes for faster inference on small devices: smaller input and alpha < 1.0
        input_shape = (224, 224, 3)  # matches our self.input_shape
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape,
            alpha=0.35  # scale down the size of the network
        )
        base_model.trainable = False  # Freeze the layers so we don't ruin pretrained weights.

        # Create new input to feed into the base model.
        inputs = tf.keras.Input(shape=input_shape)
        x = base_model(inputs)

        # This step converts the 2D feature map into a single vector by averaging.
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Add new Dense layers to learn tennis-ball-specific patterns.
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)   # Dropout helps prevent overfitting.
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # Final layer for classification: "softmax" because we have two classes.
        classification_output = tf.keras.layers.Dense(
            self.num_classes,
            activation='softmax',
            name='class_output'
        )(x)

        # Final layer for coordinates: "linear" activation because coords can be any float.
        coordinate_output = tf.keras.layers.Dense(
            4,
            activation='linear',
            name='coordinate_output'
        )(x)

        # Create the overall model.
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[classification_output, coordinate_output]
        )

        # Compile the model, telling TensorFlow which losses and metrics we want to track.
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
        """
        Trains the model on the given data, using Early Stopping to prevent overfitting.

        Args:
            X_train: Training images.
            y_train: Dictionary containing classification and coordinate labels for training.
            X_val: Validation images.
            y_val: Dictionary containing classification and coordinate labels for validation.

        Returns:
            history: A record of training metrics over each epoch.
        """
        # EarlyStopping will watch 'val_loss'. If it stops improving for 10 epochs, training stops.
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
        """
        Runs the trained model on some data and displays the results.

        - Draws circles around the true bounding boxes and predicted bounding boxes, if any.
        - Prints how long it took to do the predictions.
        - Shows the image, along with "Pred: Ball/No Ball" and "True: Ball/No Ball".

        Args:
            X_val: Array of images on which to run inference.
            y_val: The true labels for those images (both class and coordinates).
        """
        start_time = time.time()
        # "predict" returns [class_predictions, coordinate_predictions].
        predictions = self.model.predict(X_val)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")

        plt.figure(figsize=(10, 15))
        # We only show up to 30 examples to keep it manageable.
        for i in range(min(10, len(X_val))):
            plt.subplot(2, 5, i + 1)
            img = X_val[i]

            # Class predictions: pick the index with the highest probability -> 0 or 1
            pred_class = np.argmax(predictions[0][i])
            # Coordinate predictions: 4 values
            pred_coords = predictions[1][i]

            # Ground truth (true labels)
            true_class = np.argmax(y_val['class_output'][i])
            true_coords = y_val['coordinate_output'][i]

            # Convert from float [0..1] range to image [0..255] range for display with OpenCV.
            img_cv = (img * 255).astype(np.uint8)

            # If the true class is "Ball" (which we labeled as class 0), draw a blue circle.
            if true_class == 0:
                x_true, y_true, w_true, h_true = true_coords
                x_true = int(x_true * img_cv.shape[1])
                y_true = int(y_true * img_cv.shape[0])
                radius_true = int(
                    max(w_true * img_cv.shape[1],
                        h_true * img_cv.shape[0]) / 2
                )
                cv2.circle(
                    img,
                    (x_true, y_true),
                    radius_true,
                    (255, 0, 0),  # dark blue
                    2
                )

            # If the predicted class is "Ball", draw a red circle with predicted coords.
            if pred_class == 0:
                x, y, w, h = pred_coords
                x = int(x * img.shape[1])
                y = int(y * img.shape[0])
                radius = int(max(w, h) * min(img.shape[:2]) / 2)
                print(f"x {x}, y {y}, radius {radius}")
                print(f"x {img.shape[1]}, y {img.shape[0]}")
                cv2.circle(
                    img,
                    (x, y),
                    radius,
                    (0, 255, 0),  # dark red
                    2
                )

            #plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR) )
            plt.imshow(img )
            plt.title(f'Pred: {"Ball" if pred_class == 0 else "No Ball"}\n'
                      f'True: {"Ball" if true_class == 0 else "No Ball"}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def resize_images_cv2(self, images, target_size=(224, 224)):
        """
        Resizes the images to the desired target_size using OpenCV.
        This step ensures that they match the input size for our model.

        Args:
            images: A list or array of images to resize.
            target_size: The desired (width, height). Changed default to (160, 160) for speed.

        Returns:
            A NumPy array of resized images.
        """
        resized_images = [
            cv2.resize(img, target_size)  # Note OpenCV uses (width, height)
            for img in images
        ]
        return np.array(resized_images)
    
    def verify_tflite_model(self, model_path):  
        """  
        Verify the structure of the converted TFLite model  
        """  
        try:  
            # Load the TFLite model  
            interpreter = tf.lite.Interpreter(model_path=model_path)  
            interpreter.allocate_tensors()  

            # Get input and output details  
            input_details = interpreter.get_input_details()  
            output_details = interpreter.get_output_details()  

            print("\nTFLite Model Verification:")  
            print("Input Details:")  
            for i, detail in enumerate(input_details):  
                print(f"Input {i}:")  
                print(f"  Shape: {detail['shape']}")  
                print(f"  Dtype: {detail['dtype']}")  
                print(f"  Index: {detail['index']}")  

            print("\nOutput Details:")  
            for i, detail in enumerate(output_details):  
                print(f"Output {i}:")  
                print(f"  Name: {detail.get('name', 'Unknown')}")  
                print(f"  Shape: {detail['shape']}")  
                print(f"  Dtype: {detail['dtype']}")  
                print(f"  Index: {detail['index']}")  

        except Exception as e:  
            print(f"Error verifying TFLite model: {e}")  
            
    def save_tflite_model(self, filename="tennis_ball_detector.tflite"):
        """  
        Converts the current Keras model to TensorFlow Lite format  
        """  
        if self.model is None:  
            print("No model found! Train or load a model first.")  
            return  

        def representative_data_gen():  
            # Generate representative dataset for quantization  
            for _ in range(1000):  
                # Ensure input matches your model's expected input shape  
                dummy_input = tf.random.uniform([1, 224, 224, 3], 0, 1, dtype=tf.float32)  
                yield [dummy_input]  

        try:  
            @tf.function(input_signature=[  
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.uint8)  
            ])  
            def serving_default(inputs):  
                # Call the model with inputs  
                classification, coordinates = self.model(inputs)  
                return {  
                    'class_output': classification,  
                    'coordinate_output': coordinates  
                }  
             # Convert model to TFLite  
            converter = tf.lite.TFLiteConverter.from_concrete_functions([  
                serving_default.get_concrete_function()  
            ])  
            #converter = tf.lite.TFLiteConverter.from_keras_model(self.model)  
            # Quantization settings  
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  
            converter.representative_dataset = representative_data_gen  
            
            # Quantization specifics  
            converter.target_spec.supported_ops = [  
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,   
                tf.lite.OpsSet.SELECT_TF_OPS  
            ]  
            converter.target_spec.supported_types = [tf.uint8]  
            # Force float32 for conversion, then quantize  
            converter.inference_input_type = tf.float32  
            converter.inference_output_type = tf.float32

            # Convert the model  
            tflite_model = converter.convert()  

            # Save the model  
            with open(filename, "wb") as f:  
                f.write(tflite_model)  
            
            print(f"TFLite model saved to {filename}")  
            self.verify_tflite_model(filename)  
        except Exception as e:  
            print(f"Error converting model: {e}")  
            import traceback  
            traceback.print_exc()  

def main():
    """
    Main function to run the entire process:
    1) Load raw images and bounding boxes.
    2) Augment the data for variety.
    3) Shuffle and resize the data.
    4) Split into train and validation sets.
    5) Train the model.
    6) Inference and visualization on the last 30 images.

    --- ADDITIONAL NOTE FOR CORAL TPU ---
    Once this model is trained, you can export it to TFLite and perform full-integer
    quantization so it can run on the Coral Edge TPU. The rough steps are:
    
    1) Export to SavedModel:
         model.save('my_model_saved')
    2) Convert to TFLite:
         converter = tf.lite.TFLiteConverter.from_saved_model('my_model_saved')
         converter.optimizations = [tf.lite.Optimize.DEFAULT]
         # For Coral, you typically need a representative dataset and int8 input/output:
         # converter.representative_dataset = my_representative_data_gen
         # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
         # converter.inference_input_type = tf.uint8
         # converter.inference_output_type = tf.uint8
         tflite_model = converter.convert()
    3) Compile for Coral Edge TPU (using edgetpu_compiler):
         edgetpu_compiler my_model.tflite
    4) Deploy to Raspberry Pi with Coral USB and run with the PyCoral API.
    """
    # Replace these paths with the correct locations of your data.
    IMAGE_DIR = '/mnt/d/work/self_drive/tracking/data_set_ball/images'
    ANNOTATION_DIR = '/mnt/d/work/self_drive/tracking/data_set_ball/annotations'

    IMAGE_DIR_NO_BALL = '/mnt/d/work/self_drive/tracking/data_set_no_ball/images'
    ANNOTATION__NO_BALL_DIR = '/mnt/d/work/self_drive/tracking/data_set_no_ball/annotations'

    # Create an instance of our detector.
    detector = TennisBallDetector()

    # 1) Load data from disk.
    images, coordinates = detector.load_and_preprocess_data(
        IMAGE_DIR,
        ANNOTATION_DIR
    )

    images_no_ball, coordinates_no_ball = detector.load_and_preprocess_data(
        IMAGE_DIR_NO_BALL,
        ANNOTATION__NO_BALL_DIR
    )
    
    augmented_images, augmented_coordinates, augmented_validity = detector.merge_data_set(images, coordinates, images_no_ball, coordinates_no_ball)
    # 2) Augment the data for variety (brightness changes, noise, etc.).
    #augmented_images, augmented_coordinates, augmented_validity = (
    #   detector.augment_data(images, coordinates)
    #)

    #cv2.imshow('Object Detection', cv2.cvtColor(augmented_images[30], cv2.COLOR_RGB2BGR)  )
    #cv2.waitKey(100)
    
    # 3) Shuffle the augmented data so training is more random.
    dataSz = len(augmented_images)
    permutation = np.random.permutation(dataSz)
    augmented_images = augmented_images[permutation]
    augmented_coordinates = augmented_coordinates[permutation]
    augmented_validity = augmented_validity[permutation]

    # Resize images so they fit the new 160x160 input size.
    augmented_images = detector.resize_images_cv2(augmented_images)

    # 4) Split into train and validation sets, leaving 30 images for final checks.
    X_train, y_train, X_val, y_val = detector.split_data(
        augmented_images[:dataSz-10],
        augmented_validity[:dataSz-10],
        augmented_coordinates[:dataSz-10]
    )

    # 5) Create and train the model.
    detector.model = detector.create_model(len(X_train))
    detector.train(X_train, y_train, X_val, y_val)

    #cv2.imshow('Object Detection', cv2.cvtColor(augmented_images[dataSz-30], cv2.COLOR_RGB2BGR) )
    #cv2.waitKey(100)

    # 6) Use the final 30 images for inference and visualization.
    X_train, y_train, X_val, y_val = detector.split_data(
        augmented_images[dataSz-10:],
        augmented_validity[dataSz-10:],
        augmented_coordinates[dataSz-10:],
        train_split=1
    )
    detector.inference_and_visualize(X_train, y_train)
    #detector.save_tflite_model()

# This block will only run if you run the file directly, and not if you import it.
if __name__ == '__main__':
    main()
