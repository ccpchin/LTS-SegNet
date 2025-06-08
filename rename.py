import os
import pydicom

def rename_dicom_file(file_path):
    """
    Rename the DICOM file by adding '_RA64' to its filename in the same directory.
    """
    directory, filename = os.path.split(file_path)
    new_filename = f"{os.path.splitext(filename)[0]}_RA64.dcm"
    new_file_path = os.path.join(directory, new_filename)

    try:
        os.rename(file_path, new_file_path)  # Rename file in the same directory
        print(f"🔄 File renamed: {new_file_path}")
        return new_file_path
    except Exception as e:
        print(f"❌ Error renaming file: {e}")
        return None

def is_valid_dicom(file_path):
    """
    Check if a file is a valid DICOM file and if it follows RA64 Transfer Syntax.
    If not, rename the file by adding '_RA64' to the filename in the same directory.
    """
    try:
        # Open the file and check the DICOM magic number at offset 128
        with open(file_path, "rb") as f:
            f.seek(128)  # DICOM magic number location
            magic = f.read(4)
            if magic != b"DICM":
                print(f"❌ {file_path} is not a valid DICOM file (missing DICM magic number).")
                return False

        # Read the DICOM file using pydicom
        dicom_data = pydicom.dcmread(file_path)

        # Check if the Transfer Syntax UID contains RA64
        if "RA64" in str(dicom_data.file_meta.TransferSyntaxUID):
            print(f"✅ {file_path} is a valid DICOM file (RA64 detected).")
            return True
        else:
            print(f"⚠ {file_path} is a DICOM file but not in RA64 format.")

            # Rename the file in the same location
            new_file_path = rename_dicom_file(file_path)
            return False if new_file_path is None else True  # Return True if renaming was successful

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False

def process_dicom_folder(folder_path):
    """
    Scan a folder for DICOM files and rename non-RA64 files.
    """
    print(f"\n📂 Scanning folder: {folder_path}\n")
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".dcm"):
                file_path = os.path.join(root, filename)
                is_valid_dicom(file_path)

if __name__ == "__main__":
    folder_paths =[r"C:\Users\chinm\OneDrive\Desktop\3Dircadb1.1\3Dircadb1.1\LABELLED_DICOM\LABELLED_DICOM", 
              r"C:\Users\chinm\OneDrive\Desktop\3Dircadb1.1\3Dircadb1.1\PATIENT_DICOM\PATIENT_DICOM",
              r"C:\Users\chinm\OneDrive\Desktop\3Dircadb1.19\LABELLED_DICOM\LABELLED_DICOM",
              r"C:\Users\chinm\OneDrive\Desktop\3Dircadb1.19\PATIENT_DICOM\PATIENT_DICOM"
              ]

    for folder in folder_paths:
        if os.path.exists(folder):
            process_dicom_folder(folder)
        else:
            print(f"❌ Folder not found: {folder}")