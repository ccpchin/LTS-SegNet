def is_dicom_magic(file_path):
    """Check if a file has the DICOM magic number at offset 128."""
    try:
        with open(file_path, "rb") as f:
            f.seek(128)  # Move to the DICOM magic number location
            magic = f.read(4)

        if magic == b"DICM":
            print("✅ DICOM magic number found! This is a valid DICOM file.")
            return True
        else:
            print("❌ Not a valid DICOM file (wrong magic number).")
            return False

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

# Example Usage
if __name__ == "__main__":
    file_path = r"C:\Users\chinm\OneDrive\Desktop\3Dircadb1.19\PATIENT_DICOM\PATIENT_DICOM\image_0.dcm"
    is_dicom_magic(file_path)
