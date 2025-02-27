#!/bin/bash

# Function to display usage information
usage() {
  echo "Usage: $0 -r <repository> -i <image_name> -d <docker_identifier>"
  exit 1
}

# Parse command-line options
while getopts ":r:i:d:" opt; do
  case ${opt} in
    r )
      REPOSITORY="$OPTARG"
      ;;
    i )
      IMAGE_NAME="$OPTARG"
      ;;
    d )
      DOCKER_IDENTIFIER="$OPTARG"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    : )
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Check if all required arguments are provided
if [ -z "$REPOSITORY" ] || [ -z "$IMAGE_NAME" ] || [ -z "$DOCKER_IDENTIFIER" ]; then
  echo "All arguments are required."
  usage
fi

# Build the Docker image
docker build -t "${REPOSITORY}/${IMAGE_NAME}:${DOCKER_IDENTIFIER}" .

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo "Docker image ${REPOSITORY}/${IMAGE_NAME}:${DOCKER_IDENTIFIER} built successfully."
else
  echo "Failed to build Docker image."
  exit 1
fi
