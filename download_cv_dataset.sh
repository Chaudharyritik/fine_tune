#!/bin/bash
# Download Common Voice Hindi dataset from Mozilla Data Collective API

set -e  # Exit on error

cd /home/fine_tune

CLIENT_ID="mdc_b890e2403671d697a552b7132877c619"
API_KEY="c9d86b331d9125926d5b594de7c7c0b42c28d456d15f688de339d84d16ede244"
DATASET_ID="cmflnuzw5hbe47u0fvrugjyb6"

echo "="*60
echo "Downloading Common Voice Hindi Dataset"
echo "="*60

# Step 1: Create download session
echo "Step 1: Creating download session..."
RESPONSE=$(curl -s -X POST "https://datacollective.mozillafoundation.org/api/datasets/${DATASET_ID}/download" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json")

# Extract download URL (easier than token)
DOWNLOAD_URL=$(echo "$RESPONSE" | grep -o '"downloadUrl":"[^"]*' | cut -d'"' -f4)

if [ -z "$DOWNLOAD_URL" ]; then
  # Try with jq if available
  if command -v jq &> /dev/null; then
    DOWNLOAD_URL=$(echo "$RESPONSE" | jq -r '.downloadUrl // empty')
  fi
fi

if [ -z "$DOWNLOAD_URL" ]; then
  echo "ERROR: Could not extract download URL"
  echo "API Response:"
  echo "$RESPONSE"
  exit 1
fi

echo "✅ Got download URL"
echo "   File size: $(echo "$RESPONSE" | grep -o '"sizeBytes":"[^"]*' | cut -d'"' -f4) bytes"

# Step 2: Download the dataset
echo ""
echo "Step 2: Downloading dataset (this may take a while)..."
curl -L "${DOWNLOAD_URL}" \
  -H "Authorization: Bearer ${API_KEY}" \
  -o "mcv-scripted-hi-v23.0.tar.gz" \
  --progress-bar

# Verify download
if [ ! -s "mcv-scripted-hi-v23.0.tar.gz" ]; then
  echo "ERROR: Download failed or file is empty"
  exit 1
fi

FILE_SIZE=$(du -h mcv-scripted-hi-v23.0.tar.gz | cut -f1)
echo "✅ Downloaded: mcv-scripted-hi-v23.0.tar.gz (${FILE_SIZE})"

# Step 3: Extract
echo ""
echo "Step 3: Extracting dataset..."
tar -xzf mcv-scripted-hi-v23.0.tar.gz

# Step 4: Find the hi/ directory
echo ""
echo "Step 4: Locating dataset files..."
HI_PATH=$(find . -type d -name "hi" | head -1)

if [ -z "$HI_PATH" ]; then
  echo "⚠️  hi/ directory not found. Listing extracted structure:"
  find . -maxdepth 3 -type d | head -20
  exit 1
fi

# Verify required files exist
if [ ! -f "${HI_PATH}/train.tsv" ]; then
  echo "ERROR: train.tsv not found in ${HI_PATH}"
  exit 1
fi

if [ ! -f "${HI_PATH}/dev.tsv" ]; then
  echo "ERROR: dev.tsv not found in ${HI_PATH}"
  exit 1
fi

echo "✅ Dataset extracted successfully!"
echo ""
echo "Dataset location: ${HI_PATH}"
echo ""
echo "File counts:"
echo "  Train: $(wc -l < ${HI_PATH}/train.tsv) lines"
echo "  Dev:   $(wc -l < ${HI_PATH}/dev.tsv) lines"
echo "  Test:  $(wc -l < ${HI_PATH}/test.tsv 2>/dev/null || echo 'N/A') lines"
echo ""
echo "To run training, use:"
echo "  LOCAL_CV_PATH=${HI_PATH} python3 whisper_ft_local_cv.py"
echo ""
echo "Or update the default path in whisper_ft_local_cv.py to:"
echo "  LOCAL_CV_PATH = \"${HI_PATH}\""

