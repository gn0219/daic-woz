import os
import zipfile
import shutil
from tqdm import tqdm

def unzip_files(zip_dir, extract_dir):
    """
    Unzips all .zip files in the specified directory and extracts them to the target directory.
    
    Parameters:
    zip_dir (str): The directory containing the zip files.
    extract_dir (str): The directory where the files will be extracted.
    """
    os.makedirs(extract_dir, exist_ok=True)

    for file in tqdm(os.listdir(zip_dir)):
        if file.endswith(".zip"):
            zip_path = os.path.join(zip_dir, file)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            except:
                print(f"Failed: {file}")

def move_files(source_dir, destination_dir, file_extension):
    """
    Moves files with a specific extension from the source directory to the destination directory.
    
    Parameters:
    source_dir (str): The directory to search for files.
    destination_dir (str): The directory where the files will be moved.
    file_extension (str): The file extension of the files to be moved.
    """
    os.makedirs(destination_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(file_extension):
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_dir, file)
                shutil.move(source_path, destination_path)

if __name__ == "__main__":
    # Define directories
    zip_dir = "./downloads"  # Directory containing zip files
    extract_dir = "./unzipped_files"  # Directory to extract files
    wav_destination_dir = "./wav_files"  # Directory to move .wav files
    transcript_destination_dir = "./transcript_files"  # Directory to move transcript files

    # Unzip files
    unzip_files(zip_dir, extract_dir)

    # Move .wav files
    move_files(extract_dir, wav_destination_dir, ".wav")

    # Move transcript files
    move_files(extract_dir, transcript_destination_dir, "_TRANSCRIPT.csv")