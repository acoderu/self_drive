import os
import time
import cv2
import albumentations as A
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau  

################################################################################
# 1) CUSTOM LOSS FUNCTION THAT IGNORES NO-BALL COORDINATES
# ---------------------------------------------------------
# If y_true == (0,0,0,0), we set the loss to 0. Otherwise, we compute MSE.
################################################################################
#def masked_mse(y_true, y_pred):
#    """
#    y_true, y_pred: shape (batch_size, 4) => [x_center, y_center, w, h].
    # If y_true is (0,0,0,0), that means 'No Ball' → ignore by producing zero loss.
    # """
    # # Check which rows are "no-ball" by summing absolute coords:
    # # If sum(abs(coords)) == 0, then it's no-ball.
    # mask = tf.reduce_sum(tf.abs(y_true), axis=-1)  # shape (batch_size,)
    # # Convert to 1.0 if not zero (ball), 0.0 if zero (no-ball)
    # mask = tf.cast(mask > 0, tf.float32)  # shape (batch_size,)
    # # Expand to (batch_size, 1) so we can multiply elementwise with 4 coords
    # mask = tf.expand_dims(mask, axis=-1)

    # # Normal MSE: mean of (y_true - y_pred)^2
    # sq_diff = tf.square(y_true - y_pred)  # shape (batch_size, 4)
    # sq_diff_masked = sq_diff * mask       # zeroes out “no-ball” examples
    # # Sum across the 4 coords, then take mean across the batch
    # loss_per_example = tf.reduce_sum(sq_diff_masked, axis=-1)  # shape (batch_size,)
    # return tf.reduce_mean(loss_per_example)


def smooth_l1_loss(y_true, y_pred):  
    diff = tf.abs(y_true - y_pred)  
    less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)  
    loss = less_than_one * 0.5 * diff**2 + (1 - less_than_one) * (diff - 0.5)  
    return loss  

def masked_smooth_l1(y_true, y_pred):  
    mask = tf.reduce_sum(tf.abs(y_true), axis=-1)  
    mask = tf.cast(mask > 0, tf.float32)  
    mask = tf.expand_dims(mask, axis=-1)  

    smooth_l1 = smooth_l1_loss(y_true, y_pred)  # shape (batch_size, 4)  
    smooth_l1_masked = smooth_l1 * mask         # zeroes out “no-ball” examples  
    loss_per_example = tf.reduce_sum(smooth_l1_masked, axis=-1)  # shape (batch_size,)  
    return tf.reduce_mean(loss_per_example)  
################################################################################
# MAIN DETECTOR CLASS
################################################################################
class TennisBallDetector:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
        self.augmentation = A.Compose([       A.HueSaturationValue(
                hue_shift_limit=5,
                sat_shift_limit=5,
                val_shift_limit=5,
                p=0.2
            ),

            A.RandomFog(                
                alpha_coef=0.02,
                p=0.2
            ),

            A.RandomShadow(brightness_coeff=1.5, snow_point_lower=0.2, snow_point_upper=0.3, p=0.2),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.2),
            A.GaussNoise(std_range=(0.1, 0.2), p=0.2),  # 10-20% of max value
            # Use a kernel size >= 3:
            A.Blur(blur_limit=3, p=0.2),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2),
                contrast_limit=(-0.2, 0.2),
                p=0.2
            ),
        ])
    
    def load_and_preprocess_data(self, image_dir, annotation_dir):
        images = []
        coordinates = []
        if not os.path.exists(image_dir):
            return np.array([]), np.array([])
        image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            annotation_file = os.path.join(
                annotation_dir, os.path.splitext(image_file)[0] + '.txt'
            )
            try:
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if os.path.exists(annotation_file):
                    with open(annotation_file, 'r') as f:
                        annotations = f.readlines()
                    for line in annotations:
                        parts = line.strip().split()
                        height, width, _ = img.shape
                        class_id = int(float(parts[0]))
                        ymin = min(max(float(parts[1]), 0), 1) * height
                        xmin = min(max(float(parts[2]), 0), 1) * width
                        ymax = min(max(float(parts[3]), 0), 1) * height
                        xmax = min(max(float(parts[4]), 0), 1) * width

                        x_center = (xmin + xmax) / (2.0 * width)
                        y_center = (ymin + ymax) / (2.0 * height)
                        box_width = (xmax - xmin) / width
                        box_height = (ymax - ymin) / height

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

    
    def augment_data(self, images, coordinates, labels, num_augmentations=3):
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
        
        for img, coord, label in zip(images, coordinates, labels):
            for _ in range(num_augmentations):
                x_center, y_center, box_width, box_height = coord
                augmented = self.augmentation(
                            image=img,
                            bboxes=[[x_center, y_center, box_width, box_height]],
                        )
                
                # The augmented dictionary contains the transformed image.
                aug_img = augmented['image']

                # Add the augmented image and the same bounding box to our lists.
                augmented_images.append(aug_img)
                augmented_validity.append(label)
                coordinate_empty = [0,0,0,0]
                noBall = np.array([0, 1])    
                if np.array_equal(label, noBall):  
                    augmented_coordinates.append(coordinate_empty)
                else :
                    augmented_coordinates.append(coord)
                    

        return (np.array(augmented_images),
                np.array(augmented_coordinates),
                np.array(augmented_validity))
    
    ############################################################################
    # 2) FORCE NO-BALL COORDINATES TO (0,0,0,0)
    ############################################################################
    def merge_data_set(self, images, coordinates,
                       images_no_ball, coordinates_no_ball):
        """
        - For Ball images: label=[1, 0].
        - For No-Ball images: label=[0, 1] & coords forced to (0,0,0,0).
        """
        # All ball images
        ball_label = [1, 0]
        ball_labels = [ball_label for _ in range(len(images))]
        merged_images = list(images)
        merged_coordinates = list(coordinates)
        merged_validity = list(ball_labels)

        # All no-ball images => Force coords to (0,0,0,0)
        no_ball_label = [0, 1]
        no_ball_labels = [no_ball_label for _ in range(len(images_no_ball))]
        images_no_ball = list(images_no_ball)
        # Overwrite all bounding boxes for no-ball
        coordinates_no_ball = np.zeros_like(coordinates_no_ball)
        # or just do coordinates_no_ball[:,:] = 0

        merged_images.extend(images_no_ball)
        merged_coordinates.extend(coordinates_no_ball)
        merged_validity.extend(no_ball_labels)

        return (np.array(merged_images),
                np.array(merged_coordinates),
                np.array(merged_validity))

    def split_data(self, images, labels, coordinates, train_split=0.9):
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
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape,
            alpha=0.35
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=self.input_shape)
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
        

        ############################################################################
        # 3) INCREASE COORDINATE LOSS WEIGHT & USE masked_mse
        ############################################################################
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss={
                'class_output': 'categorical_crossentropy',
                'coordinate_output': masked_smooth_l1  # custom masked MSE
            },
            loss_weights={
                'class_output': 1.0,
                'coordinate_output': 5.0  # increase to emphasize bounding-box accuracy
            },
            metrics={
                'class_output': 'accuracy',
                'coordinate_output': 'mse'
            }
        )
        return model

    def train(self, X_train, y_train, X_val, y_val):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True
        )
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
            batch_size=32,
            callbacks=[early_stopping, reduce_lr]
        )
        return history

    def inference_and_visualize(self, X_val, y_val):
        start_time = time.time()
        predictions = self.model.predict(X_val)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")

        plt.figure(figsize=(30, 15))
        for i in range(min(30, len(X_val))):
            plt.subplot(6, 5, i + 1)
            img = X_val[i]
            pred_class = np.argmax(predictions[0][i])
            pred_coords = predictions[1][i]

            true_class = np.argmax(y_val['class_output'][i])
            true_coords = y_val['coordinate_output'][i]

            img_cv = (img * 255).astype(np.uint8)

            # Draw ground-truth if it's a ball
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
                    (255, 0, 0),
                    2
                )

            # Draw predicted circle if predicted class is “Ball”
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
            plt.title(f'Pred: {"Ball" if pred_class == 0 else "No Ball"}\n'
                      f'True: {"Ball" if true_class == 0 else "No Ball"}')
            plt.axis('off')

        plt.tight_layout()
        plt.show()

    def resize_images_cv2(self, images, target_size=(224, 224)):
        resized_images = [
            cv2.resize(img, target_size)
            for img in images
        ]
        # Scale to [0..1] if desired (helpful if your MobileNet expects 0..1 or 0..255)
        return np.array(resized_images) / 255.0

    def verify_tflite_model(self, model_path):
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

    def save_tflite_model(self, filename="tennis_ball_detector_2.tflite"):
        if self.model is None:
            print("No model found! Train or load a model first.")
            return

        def representative_data_gen():
            for _ in range(1000):
                dummy_input = tf.random.uniform([1, 224, 224, 3], 0, 1, dtype=tf.float32)
                yield [dummy_input]

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

            converter = tf.lite.TFLiteConverter.from_concrete_functions([
                serving_default.get_concrete_function()
            ])
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            converter.target_spec.supported_types = [tf.uint8]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32

            tflite_model = converter.convert()

            with open(filename, "wb") as f:
                f.write(tflite_model)

            print(f"TFLite model saved to {filename}")
            self.verify_tflite_model(filename)

        except Exception as e:
            print(f"Error converting model: {e}")
            import traceback
            traceback.print_exc()

##############################################################################
# MAIN
##############################################################################
def main():
    # Replace these paths with the correct locations of your data.
    IMAGE_DIR = '/mnt/d/work/self_drive/tracking/data_set_ball/images'
    ANNOTATION_DIR = '/mnt/d/work/self_drive/tracking/data_set_ball/annotations'

    IMAGE_DIR_NO_BALL = '/mnt/d/work/self_drive/tracking/data_set_no_ball/images'
    ANNOTATION_NO_BALL_DIR = '/mnt/d/work/self_drive/tracking/data_set_no_ball/annotations'

    detector = TennisBallDetector()

    # 1) Load data
    images, coordinates = detector.load_and_preprocess_data(IMAGE_DIR, ANNOTATION_DIR)
    images_no_ball, coordinates_no_ball = detector.load_and_preprocess_data(
        IMAGE_DIR_NO_BALL, ANNOTATION_NO_BALL_DIR
    )

    # 2) Merge dataset => “Ball” + “No Ball” (set no-ball coords to 0,0,0,0)
    merged_images, merged_coordinates, merged_validity = detector.merge_data_set(
        images, coordinates, images_no_ball, coordinates_no_ball
    )

    merged_images, merged_coordinates, merged_validity = detector.augment_data(merged_images, 
                                                                               merged_coordinates, merged_validity,  num_augmentations=20)

    # Shuffle
    dataSz = len(merged_images)
    permutation = np.random.permutation(dataSz)
    merged_images = merged_images[permutation]
    merged_coordinates = merged_coordinates[permutation]
    merged_validity = merged_validity[permutation]

    # 3) Resize & scale images
    merged_images = detector.resize_images_cv2(merged_images)  # shape => (224,224), scaled

    # 4) Split train vs validation
    #  -- Reserve last 10 for final check
    X_train, y_train, X_val, y_val = detector.split_data(
        merged_images[:dataSz],
        merged_validity[:dataSz],
        merged_coordinates[:dataSz]
    )

    # 5) Create & train
    detector.model = detector.create_model(len(X_train))
    detector.train(X_train, y_train, X_val, y_val)

    # 6) Final check on last 10 images
    # X_inf, y_inf_class, _, _ = detector.split_data(
    #     merged_images[dataSz-30:],
    #     merged_validity[dataSz-30:],
    #     merged_coordinates[dataSz-30:],
    #     train_split=1  # everything as “training portion”
    # )
    #detector.inference_and_visualize(X_inf, y_inf_class)

    # Optional: Save TFLite
    detector.save_tflite_model()

if __name__ == '__main__':
    main()
