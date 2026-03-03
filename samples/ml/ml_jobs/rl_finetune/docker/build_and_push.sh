#!/bin/bash
# Build and push fresh AReaL image to Snowflake registry
# Run this on a machine with Docker access

set -e

# Configuration
REGISTRY="preprod8-notebook-mltest.awsuswest2preprod8.registry-dev.snowflakecomputing.com"
REPO="rl_training_db/rl_schema/rl_images"
IMAGE_NAME="areal-fresh"
TAG="v1"

FULL_IMAGE="${REGISTRY}/${REPO}/${IMAGE_NAME}:${TAG}"

echo "=== Building fresh AReaL image ==="
echo "Target: ${FULL_IMAGE}"
echo ""

cd "$(dirname "$0")"

# Build the image
echo "Building image..."
docker build -f Dockerfile.fresh -t ${IMAGE_NAME}:${TAG} .

# Tag for registry
echo "Tagging for registry..."
docker tag ${IMAGE_NAME}:${TAG} ${FULL_IMAGE}

# Login to Snowflake registry
echo ""
echo "=== Login to Snowflake registry ==="
echo "Run: snow spcs image-registry login"
echo "Or manually: docker login ${REGISTRY}"
echo ""
read -p "Press Enter after logging in to continue with push..."

# Push to registry
echo "Pushing to registry..."
docker push ${FULL_IMAGE}

echo ""
echo "=== Done! ==="
echo "Image pushed: ${FULL_IMAGE}"
echo ""
echo "Update your service spec to use this image:"
echo "  image: ${FULL_IMAGE}"
