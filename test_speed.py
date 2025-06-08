import numpy as np
import pydicom
import cv2
import time
from tensorflow.keras.models import load_model

# Load model
model = load_model("segnet_liver_tumor.keras")

# Preprocess and validate DICOM + measure speed
def preprocess_dicom_with_speed(file_path):
    try:
        dicom = pydicom.dcmread(file_path)

        # Ensure DICOM contains pixel data
        if not hasattr(dicom, "pixel_array"):
            raise ValueError("❌ DICOM file does not contain pixel data!")

        image = dicom.pixel_array.astype(np.float32)

        print(f"📂 DICOM file loaded: {file_path}")
        print(f"📏 Image shape before resize: {image.shape}")

        if image.size == 0:
            raise ValueError(f"❌ Error: Image is empty or invalid! Shape: {image.shape}")

        # Normalize
        max_val = np.max(image)
        print(f"🛠 Image max value: {max_val}")
        image = image / max_val if max_val > 0 else image

        # Resize to (512, 512)
        resized_image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

        # Expand to (1, 512, 512, 1)
        processed_image = np.expand_dims(resized_image, axis=-1)
        processed_image = np.expand_dims(processed_image, axis=0)

        print(f"✅ Preprocessed image shape: {processed_image.shape}")

        # Measure inference time
        start_time = time.time()
        _ = model.predict(processed_image)
        duration = time.time() - start_time
        print(f"✅ Model inference time: {duration:.4f} seconds")

        return processed_image

    except Exception as e:
        print(f"❌ Error in preprocess_dicom_with_speed: {e}")
        return None

# Example usage:
test_file = "uploads/image_14_RA64.dcm"  # Replace with a valid DICOM path
preprocess_dicom_with_speed(test_file)
