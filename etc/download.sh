#!/bin/bash

# Base URL where the files are hosted
BASE_URL="download_url" # Replace with the actual URL

# Directory to save downloaded files
DOWNLOAD_DIR="./downloads"
mkdir -p "$DOWNLOAD_DIR"

# Loop through file numbers 300 to 492
for i in {300..492}; do

    FILE_NAME="${i}_P.zip"
    URL="$BASE_URL/$FILE_NAME"
    echo "Downloading $FILE_NAME..."
    wget -P "$DOWNLOAD_DIR" "$URL"

    if [ $? -eq 0 ]; then
        echo "$FILE_NAME downloaded successfully."
    else
        echo "Failed to download $FILE_NAME."
    fi
    
done

echo "Download process completed."