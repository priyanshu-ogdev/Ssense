#!/bin/bash
set -e

echo "🖼️ Processing Ssense extension icons..."

SOURCE_IMG="$HOME/Downloads/a.png"
DEST_DIR="apps/extension/public/icons"

# 1. Verify source exists
if [ ! -f "$SOURCE_IMG" ]; then
    echo "❌ Error: Source image not found at $SOURCE_IMG"
    exit 1
fi

# 2. Ensure ImageMagick is installed
if ! command -v convert &> /dev/null; then
    echo "⚠️ ImageMagick is required but not installed."
    echo "Installing now..."
    sudo apt-get update && sudo apt-get install imagemagick -y
fi

# 3. Create destination directory
mkdir -p "$DEST_DIR"

echo "✂️ Center-cropping and resizing to strict Chrome dimensions..."

# Chrome Web Store & Installation Icon (128x128)
convert "$SOURCE_IMG" -resize 128x128^ -gravity center -extent 128x128 "$DEST_DIR/icon128.png"
echo "✅ Generated icon128.png"

# Extension Management Page Icon (48x48)
convert "$SOURCE_IMG" -resize 48x48^ -gravity center -extent 48x48 "$DEST_DIR/icon48.png"
echo "✅ Generated icon48.png"

# Browser Toolbar Favicon (16x16)
convert "$SOURCE_IMG" -resize 16x16^ -gravity center -extent 16x16 "$DEST_DIR/icon16.png"
echo "✅ Generated icon16.png"

echo ""
echo "🎉 Icon processing complete. Your Chrome Extension is fully loaded."