#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Physics Engine Preview Video Generator ===${NC}\n"

# Check for clean working directory
if [[ -n $(git status --porcelain) ]]; then
    echo -e "${RED}Error: Working directory is not clean${NC}"
    echo "Please commit or stash your changes before running this script"
    exit 1
fi

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: ffmpeg is not installed${NC}"
    echo "Please install ffmpeg to use video recording functionality"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
    exit 1
fi

# Store current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo -e "${GREEN}Current branch: $CURRENT_BRANCH${NC}"

# Generate test data
echo -e "\n${BLUE}Generating test data...${NC}"
cd tests
python3 create_test_dump.py
cd ..

# Record video (5 seconds at 30fps = 150 frames)
echo -e "\n${BLUE}Recording video...${NC}"
cargo run --features viz --release --bin debug_viz -- \
    --oracle tests/oracle_dump.npy \
    --record preview.mp4 \
    --duration 5 \
    --fps 30

# Check if video was created
if [ ! -f preview.mp4 ]; then
    echo -e "${RED}Error: Video recording failed${NC}"
    exit 1
fi

# Get video size
VIDEO_SIZE=$(du -h preview.mp4 | cut -f1)
echo -e "${GREEN}Video created: preview.mp4 (${VIDEO_SIZE})${NC}"

# Switch to preview branch
echo -e "\n${BLUE}Switching to preview branch...${NC}"
if git show-ref --verify --quiet refs/heads/preview; then
    git checkout preview
else
    git checkout --orphan preview
fi

# Clear all files
git rm -rf . 2>/dev/null || true
git clean -fdx

# Add and commit video
git add preview.mp4
git commit -m "Update preview video"

# Force push
echo -e "\n${BLUE}Pushing to remote...${NC}"
git push -f origin preview

# Return to original branch
echo -e "\n${BLUE}Returning to $CURRENT_BRANCH...${NC}"
git checkout $CURRENT_BRANCH

# Clean up
rm -f preview.mp4

echo -e "\n${GREEN}✅ Preview updated successfully!${NC}"
echo -e "Video is now available at:"
echo -e "  https://raw.githubusercontent.com/<USER>/<REPO>/preview/preview.mp4"
echo -e "\nDon't forget to update <USER> and <REPO> in your README.md!"