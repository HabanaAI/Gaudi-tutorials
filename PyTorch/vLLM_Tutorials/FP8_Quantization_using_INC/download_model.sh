# Check if a model name is provided as an argument
if [ $# -ne 1 ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

# Get the model name from the argument
MODEL_NAME="$1"

# Construct the checkpoint path
CHECKPOINT_PATH="/root/models/$MODEL_NAME/"

# Update package lists and install git-lfs
apt-get update
apt-get install -y git-lfs

# Install git-lfs
git lfs install

# Clone the repository
git clone "https://huggingface.co/$MODEL_NAME" "$CHECKPOINT_PATH"

echo "Model $MODEL_NAME cloned to $CHECKPOINT_PATH"
