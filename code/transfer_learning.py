#!/usr/bin/env python3
"""
transfer_learning.py
--------------------
This script demonstrates a transfer-learning approach using a MobileNetV2 backbone.
The model classifies whether an image contains a tennis ball ("Ball") or not ("No Ball"),
and also estimates bounding box coordinates for the ball.

Key steps include:
1. Data loading & preprocessing (both "Ball" and "No Ball" datasets).
2. Data augmentation using albumentations.
3. Merging "Ball" and "No Ball" data into one dataset.
4. Splitting data into training/validation sets.
5. Building a MobileNetV2-based model for classification + coordinate regression.
6. Training, evaluation, and optional TFLite conversion.
"""

import os
# Environment variable to set modern Keras usage
os.environ['TF_USE_LEGACY_KERAS'] = '0'

import time
import cv2
import albumentations as A
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Keras callbacks for learning rate management
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler

import math
# import tensorflow_model_optimization as tfmot  # Only if you plan to use TF model optimization

# -------------------------
# Utility Functions
# -------------------------
def cosine_annealing(epoch, lr):
    """
    Adjust learning rate based on the cosine annealing formula.

    Args:
        epoch (int): Current epoch number.
        lr (float): Current learning rate.

    Returns:
        float: Adjusted learning rate according to cosine annealing.
    """
    epochs = 50  # Total number of training epochs to assume
    cosine = math.cos(epoch / epochs * math.pi)
    return float(0.5 * (1 + cosine) * lr)


def smooth_l1_loss(y_true, y_pred):
    """
    Smooth L1 Loss (Huber-like) for bounding box regression.

    Args:
        y_true (Tensor): Ground-truth bounding box coordinates.
        y_pred (Tensor): Predicted bounding box coordinates.

    Returns:
        Tensor: Smooth L1 loss value.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)
    loss = less_than_one * 0.5 * diff**2 + (1 - less_than_one) * (diff - 0.5)
    return loss


def masked_smooth_l1(y_true, y_pred):
    """
    Apply smooth L1 loss only to examples that actually contain a ball.

    Args:
        y_true (Tensor): Ground-truth bounding box coords (or zeros if no ball).
        y_pred (Tensor): Predicted bounding box coords.

    Returns:
        Tensor: Average Smooth L1 loss across non-empty ground-truths.
    """
    # Compute sum of absolute values (to check if coords are non-zero)
    mask = tf.reduce_sum(tf.abs(y_true), axis=-1)
    mask = tf.cast(mask > 0, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)

    smooth_l1 = smooth_l1_loss(y_true, y_pred)  # shape (batch_size, 4)
    smooth_l1_masked = smooth_l1 * mask         # zero out examples with no ball
    loss_per_example = tf.reduce_sum(smooth_l1_masked, axis=-1)  # shape (batch_size,)
    return tf.reduce_mean(loss_per_example)


# -----------------------------------------------------------------------------
# CLASS: TennisBallDetector
# -----------------------------------------------------------------------------
class TennisBallDetector:
    """
    A class that handles:
      - Data loading & preprocessing
      - Data augmentation
      - Model creation (MobileNetV2 backbone)
      - Training & evaluation
      - Optional TFLite conversion
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        """
        Args:
            input_shape (tuple): Shape of input images (height, width, channels).
            num_classes (int): Number of output classes. In this case:
                               "Ball" (index=0) and "No Ball" (index=1).
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

        # Define a set of Albumentations transformations for data augmentation
        self.augmentation = A.Compose([
            A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=5,
                val_shift_limit=5,
                p=0.2
            ),
            A.RandomFog(
                alpha_coef=0.02,
                p=0.2
            ),
            A.RandomShadow(
                brightness_coeff=1.5,
                snow_point_lower=0.2,
                snow_point_upper=0.3,
                p=0.2
            ),
            A.RandomShadow(
                num_shadows_lower=1,
                num_shadows_upper=1,
                shadow_dimension=5,
                shadow_roi=(0, 0.5, 1, 1),
                p=0.2
            ),
            A.GaussNoise(
                std_range=(0.1, 0.2),
                p=0.2
            ),
            # Use a kernel size >= 3
            A.Blur(
                blur_limit=3,
                p=0.2
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.2
            ),
        ])

    # -------------------------
    # Data Loading & Preprocessing
    # -------------------------
    def load_and_preprocess_data(self, image_dir, annotation_dir):
        """
        Load images and their bounding box annotations from disk.
        Converts bounding boxes from absolute coords to normalized [x_center, y_center, w, h].

        Args:
            image_dir (str): Path to the directory containing images.
            annotation_dir (str): Path to the directory containing .txt annotation files.

        Returns:
            tuple: (images array, coordinates array)
                   images are resized & converted to RGB,
                   coordinates is an array of normalized bounding boxes.
        """
        images = []
        coordinates = []

        # If there's no directory, we return empty arrays
        if not os.path.exists(image_dir):
            return np.array([]), np.array([])

        # List valid image files
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            annotation_file = os.path.join(
                annotation_dir,
                os.path.splitext(image_file)[0] + '.txt'
            )

            try:
                # Read & convert image
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Read annotation if it exists
                if os.path.exists(annotation_file):
                    with open(annotation_file, 'r') as f:
                        annotations = f.readlines()

                    for line in annotations:
                        parts = line.strip().split()

                        # Extract bounding box info & clamp to [0..1]
                        height, width, _ = img.shape
                        class_id = int(float(parts[0]))  # not used directly here
                        ymin = min(max(float(parts[1]), 0), 1) * height
                        xmin = min(max(float(parts[2]), 0), 1) * width
                        ymax = min(max(float(parts[3]), 0), 1) * height
                        xmax = min(max(float(parts[4]), 0), 1) * width

                        # Convert to normalized center-based coordinates
                        x_center = (xmin + xmax) / (2.0 * width)
                        y_center = (ymin + ymax) / (2.0 * height)
                        box_width = (xmax - xmin) / width
                        box_height = (ymax - ymin) / height

                        # Resize the image to the model's input dimensions
                        resized_img = cv2.resize(
                            img, (self.input_shape[1], self.input_shape[0])
                        )
                        images.append(resized_img)
                        coordinates.append([
                            x_center, y_center, box_width, box_height
                        ])

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

        return np.array(images), np.array(coordinates)

    # -------------------------
    # Data Augmentation
    # -------------------------
    def augment_data(self, images, coordinates, labels, num_augmentations=3):
        """
        Apply random data augmentations to each image-bbox pair.

        Args:
            images (array): Array of images.
            coordinates (array): Array of bounding boxes [x_center, y_center, w, h].
            labels (array): One-hot labels [1,0]=Ball, [0,1]=No Ball.
            num_augmentations (int): How many times to augment each image.

        Returns:
            (augmented_images, augmented_coordinates, augmented_validity):
            All as NumPy arrays.
        """
        augmented_images = []
        augmented_coordinates = []
        augmented_validity = []

        # For each image, do a certain number of augmentations
        # Keep the bounding box the same, but apply transformations to the image.
        for img, coord, label in zip(images, coordinates, labels):
            for _ in range(num_augmentations):
                x_center, y_center, box_width, box_height = coord

                # Albumentations can track bounding boxes, but here we keep them the same
                augmented = self.augmentation(
                    image=img,
                    bboxes=[[x_center, y_center, box_width, box_height]],
                )

                aug_img = augmented['image']  # The augmented image

                # Store the result
                augmented_images.append(aug_img)
                augmented_validity.append(label)

                # If the label is "No Ball", then we zero the coordinates
                # since there's no real bounding box
                no_ball = np.array([0, 1])
                if np.array_equal(label, no_ball):
                    coordinate_empty = [0, 0, 0, 0]
                    augmented_coordinates.append(coordinate_empty)
                else:
                    augmented_coordinates.append(coord)

        return (
            np.array(augmented_images),
            np.array(augmented_coordinates),
            np.array(augmented_validity)
        )

    # -------------------------
    # Merge Datasets
    # -------------------------
    def merge_data_set(self, images, coordinates, images_no_ball, coordinates_no_ball):
        """
        Merge "Ball" and "No Ball" data.

        For ball images => label=[1,0].
        For no-ball images => label=[0,1] and coords forced to (0,0,0,0).

        Args:
            images (array): "Ball" images.
            coordinates (array): bounding boxes for "Ball" images.
            images_no_ball (array): "No Ball" images.
            coordinates_no_ball (array): bounding boxes (will be zeroed).

        Returns:
            (merged_images, merged_coordinates, merged_validity): merged arrays.
        """
        # "Ball" label
        ball_label = [1, 0]
        ball_labels = [ball_label for _ in range(len(images))]

        merged_images = list(images)
        merged_coordinates = list(coordinates)
        merged_validity = list(ball_labels)

        # "No Ball" label
        no_ball_label = [0, 1]
        no_ball_labels = [no_ball_label for _ in range(len(images_no_ball))]
        images_no_ball = list(images_no_ball)

        # Zero out coords for no-ball
        coordinates_no_ball = np.zeros_like(coordinates_no_ball)

        # Extend merges
        merged_images.extend(images_no_ball)
        merged_coordinates.extend(coordinates_no_ball)
        merged_validity.extend(no_ball_labels)

        return (
            np.array(merged_images),
            np.array(merged_coordinates),
            np.array(merged_validity)
        )

    # -------------------------
    # Train/Validation Split
    # -------------------------
    def split_data(self, images, labels, coordinates, train_split=0.9):
        """
        Split images/labels/coordinates into training and validation sets.

        Args:
            images (array): The images to split.
            labels (array): One-hot labels.
            coordinates (array): bounding boxes.
            train_split (float): Proportion of data for training.

        Returns:
            X_train, y_train, X_val, y_val (dict forms for y).
        """
        split_idx = int(len(images) * train_split)

        X_train = images[:split_idx]
        y_train_class = labels[:split_idx]
        y_train_coord = coordinates[:split_idx]

        X_val = images[split_idx:]
        y_val_class = labels[split_idx:]
        y_val_coord = coordinates[split_idx:]

        return (
            X_train,
            {
                'class_output': y_train_class,
                'coordinate_output': y_train_coord
            },
            X_val,
            {
                'class_output': y_val_class,
                'coordinate_output': y_val_coord
            },
        )

    # -------------------------
    # Model Creation
    # -------------------------
    def create_model(self):
        """
        Builds a MobileNetV2-based model that outputs both:
         - class_output: a 2D classification (Ball vs. No Ball)
         - coordinate_output: bounding box coords (4D).

        Returns:
            tf.keras.Model
        """
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            alpha=0.35  # Smaller version of MobileNet
        )

        # Set most layers as non-trainable, unfreeze only the last few
        base_model.trainable = True
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        # Start building the model
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs)

        # Extra convolution & pooling
        x = tf.keras.layers.Conv2D(
            256, (3, 3),
            activation='relu6',
            padding='same'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)  # Replaces GlobalAveragePooling2D

        # Classification branch
        x_class = tf.keras.layers.Dense(256, activation='relu6')(x)
        x_class = tf.keras.layers.Dropout(0.3)(x_class)
        classification_output = tf.keras.layers.Dense(
            self.num_classes,
            activation='softmax',
            name='class_output'
        )(x_class)

        # Coordinate regression branch
        x_loc = tf.keras.layers.Dense(128, activation='relu6')(x)
        x_loc = tf.keras.layers.Dropout(0.3)(x_loc)
        coordinate_output = tf.keras.layers.Dense(
            4,
            activation='linear',
            name='coordinate_output'
        )(x_loc)

        # Combine into a Model with two outputs
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[classification_output, coordinate_output]
        )

        # Compile the model with two different loss functions
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss={
                'class_output': 'categorical_crossentropy',
                'coordinate_output': tf.keras.losses.Huber()  # or smooth L1
            },
            loss_weights={
                'class_output': 1.0,
                'coordinate_output': 5.0  # weigh bounding box regression more
            },
            metrics={
                'class_output': 'accuracy',
                'coordinate_output': 'mse'
            }
        )
        return model

    # -------------------------
    # Training
    # -------------------------
    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model with EarlyStopping and learning rate reduction.

        Args:
            X_train (array): Training images.
            y_train (dict): {'class_output': one-hot labels,
                             'coordinate_output': bounding box coords}
            X_val (array): Validation images.
            y_val (dict):  {'class_output': one-hot labels,
                             'coordinate_output': bounding box coords}

        Returns:
            history (History object): Training history.
        """
        # Early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True
        )

        # Reduce LR on plateau callback
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
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
            batch_size=64,
            callbacks=[early_stopping, reduce_lr]
        )
        return history

    # -------------------------
    # Inference & Visualization
    # -------------------------
    def inference_and_visualize(self, X_val, y_val):
        """
        Run model inference on validation data, then visualize predictions vs ground truth.

        Args:
            X_val (array): Validation images
            y_val (dict): Validation labels & coords
        """
        start_time = time.time()
        predictions = self.model.predict(X_val)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")

        plt.figure(figsize=(30, 15))

        # Show up to 30 examples
        for i in range(min(30, len(X_val))):
            plt.subplot(6, 5, i + 1)
            img = X_val[i]

            # Model outputs: [class_output, coordinate_output]
            pred_class = np.argmax(predictions[0][i])      # Argmax of classification
            pred_coords = predictions[1][i]               # 4D box coords

            true_class = np.argmax(y_val['class_output'][i])  # Argmax of ground-truth class
            true_coords = y_val['coordinate_output'][i]       # 4D ground-truth box

            # Convert to uint8 image for drawing
            img_cv = (img * 255).astype(np.uint8)

            # Draw ground-truth circle if it's a ball
            if true_class == 0:
                x_true, y_true, w_true, h_true = true_coords
                x_true = int(x_true * img_cv.shape[1])
                y_true = int(y_true * img_cv.shape[0])
                radius_true = int(
                    max(w_true * img_cv.shape[1], h_true * img_cv.shape[0]) / 2
                )
                cv2.circle(
                    img,
                    (x_true, y_true),
                    radius_true,
                    (255, 0, 0),
                    2
                )

            # Draw predicted circle if model predicted "Ball"
            if pred_class == 0:
                x, y, w, h = pred_coords
                x = int(x * img.shape[1])
                y = int(y * img.shape[0])
                radius = int(max(w, h) * min(img.shape[:2]) / 2)
                cv2.circle(
                    img,
                    (x, y),
                    radius,
                    (0, 255, 0),
                    2
                )

            plt.imshow(img)
            plt.title(
                f'Pred: {"Ball" if pred_class == 0 else "No Ball"}\n'
                f'True: {"Ball" if true_class == 0 else "No Ball"}'
            )
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    # -------------------------
    # Image Resizing
    # -------------------------
    def resize_images_cv2(self, images, target_size=(224, 224)):
        """
        Resizes and normalizes a batch of images.

        Args:
            images (array): Original images of varying sizes.
            target_size (tuple): Desired width & height (e.g., (224, 224)).

        Returns:
            array: Resized and normalized images (pixel values in [0..1]).
        """
        resized_images = [
            cv2.resize(img, target_size)
            for img in images
        ]
        return np.array(resized_images) / 255.0

    # -------------------------
    # TFLite Model Verification
    # -------------------------
    def verify_tflite_model(self, model_path):
        """
        Inspect the structure & I/O details of a converted TFLite model.

        Args:
            model_path (str): Path to the .tflite file
        """
        try:
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

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

    # -------------------------
    # TFLite Conversion
    # -------------------------
    def save_tflite_model(self, images, filename="tennis_ball_detector_1.tflite"):
        """
        Convert the trained Keras model to an INT8 TFLite model, using a representative dataset.

        Args:
            images (array): Representative images (for calibration).
            filename (str): Output TFLite filename.
        """
        if self.model is None:
            print("No model found! Train or load a model first.")
            return

        def preprocess_image(image):
            """
            Helper to scale and convert an image to uint8 before TFLite conversion.
            """
            input_data = (image * 255).astype(np.uint8)  # scale to [0..255]
            input_data = np.expand_dims(input_data, axis=0)  # add batch dimension
            return input_data

        def representative_data_gen():
            # Generate data for calibration from each image in 'images'
            for image in images:
                preprocessed_image = preprocess_image(image)
                yield [preprocessed_image]

        try:
            @tf.function(input_signature=[
                tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.uint8)
            ])
            def serving_default(inputs):
                classification, coordinates = self.model(inputs)
                return {
                    'class_output': classification,
                    'coordinate_output': coordinates
                }

            # Create a TFLite converter from the concrete function
            converter = tf.lite.TFLiteConverter.from_concrete_functions([
                serving_default.get_concrete_function()
            ])

            # Optimize and quantize to INT8
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.float32

            tflite_model = converter.convert()

            # Save .tflite file
            with open(filename, "wb") as f:
                f.write(tflite_model)

            print(f"TFLite model saved to {filename}")
            self.verify_tflite_model(filename)

        except Exception as e:
            print(f"Error converting model: {e}")
            import traceback
            traceback.print_exc()


# -----------------------------------------------------------------------------
# MAIN FUNCTION
# -----------------------------------------------------------------------------
def main():
    """
    Main pipeline for creating and training the tennis ball detector.
    Also demonstrates merging "Ball" and "No Ball" data,
    augmenting, splitting, training, and TFLite export.
    """
    # Paths to your data. Please adjust these paths as needed.
    IMAGE_DIR = '/mnt/d/work/self_drive/tracking/data_set_ball/images'
    ANNOTATION_DIR = '/mnt/d/work/self_drive/tracking/data_set_ball/annotations'

    IMAGE_DIR_NO_BALL = '/mnt/d/work/self_drive/tracking/data_set_no_ball/images'
    ANNOTATION_NO_BALL_DIR = '/mnt/d/work/self_drive/tracking/data_set_no_ball/annotations'

    detector = TennisBallDetector()

    # 1) Load "Ball" dataset
    images, coordinates = detector.load_and_preprocess_data(IMAGE_DIR, ANNOTATION_DIR)

    # 2) Load "No Ball" dataset
    images_no_ball, coordinates_no_ball = detector.load_and_preprocess_data(
        IMAGE_DIR_NO_BALL, ANNOTATION_NO_BALL_DIR
    )

    # 3) Merge datasets => Ball + No Ball
    merged_images, merged_coordinates, merged_validity = detector.merge_data_set(
        images, coordinates, images_no_ball, coordinates_no_ball
    )

    # 4) Augment data
    merged_images, merged_coordinates, merged_validity = detector.augment_data(
        merged_images,
        merged_coordinates,
        merged_validity,
        num_augmentations=20
    )

    # 5) Shuffle entire dataset
    dataSz = len(merged_images)
    permutation = np.random.permutation(dataSz)
    merged_images = merged_images[permutation]
    merged_coordinates = merged_coordinates[permutation]
    merged_validity = merged_validity[permutation]

    # 6) Resize images to (224,224) and normalize
    merged_images = detector.resize_images_cv2(merged_images)

    # 7) Train/Val split
    X_train, y_train, X_val, y_val = detector.split_data(
        merged_images[:dataSz],
        merged_validity[:dataSz],
        merged_coordinates[:dataSz]
    )

    # 8) Create and Train the model
    detector.model = detector.create_model()
    detector.train(X_train, y_train, X_val, y_val)

    # 9) (Optional) Inference + Visualization on a subset
    #    Uncomment if you want to see predictions:
    # X_inf, y_inf_class, _, _ = detector.split_data(
    #     merged_images[dataSz - 30:],
    #     merged_validity[dataSz - 30:],
    #     merged_coordinates[dataSz - 30:],
    #     train_split=1
    # )
    # detector.inference_and_visualize(X_inf, y_inf_class)

    # 10) Save as TFLite
    detector.save_tflite_model(merged_images)


if __name__ == '__main__':
    main()
