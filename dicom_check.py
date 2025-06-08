import os
import pydicom

def is_valid_dicom(file_path):
    """
    Check if a file is a valid DICOM file and if it follows RA64 Transfer Syntax.
    If not, rename the file by adding '_RA64' to the filename.
    """
    try:
        # Open the file and check the DICOM magic number at offset 128
        with open(file_path, "rb") as f:
            f.seek(128)  # DICOM magic number location
            magic = f.read(4)
            if magic != b"DICM":
                print("❌ Not a valid DICOM file (missing DICM magic number).")
                return False

        # Read the DICOM file using pydicom
        dicom_data = pydicom.dcmread(file_path)

        # Check if the Transfer Syntax UID contains RA64
        if "RA64" in str(dicom_data.file_meta.TransferSyntaxUID):
            print("✅ Valid DICOM file (RA64 Transfer Syntax detected).")
            return True
        else:
            print("⚠ DICOM file found, but not in RA64 format.")

            # Rename the file to include "_RA64"
            new_file_path = rename_dicom_file(file_path)
            print(f"🔄 File renamed to: {new_file_path}")
            return False  # File is DICOM, but not RA64

    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return False

def rename_dicom_file(file_path):
    """
    Rename the DICOM file by adding '_RA64' to its filename.
    """
    directory, filename = os.path.split(file_path)
    new_filename = f"{os.path.splitext(filename)[0]}_RA64.dcm"
    new_file_path = os.path.join(directory, new_filename)
    
    os.rename(file_path, new_file_path)  # Rename the file
    return new_file_path

if __name__ == "__main__":
    file_path = r"C:\Users\chinm\OneDrive\Desktop\3Dircadb1.19\PATIENT_DICOM\PATIENT_DICOM\image_0.dcm"
    is_valid_dicom(file_path)
