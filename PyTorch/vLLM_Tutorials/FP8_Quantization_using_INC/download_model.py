import os
from huggingface_hub import snapshot_download

snapshot_download(repo_id=os.environ['MODEL_NAME'], local_dir=os.environ['CHECKPOINT_PATH'], local_dir_use_symlinks=False, token=os.environ['HF_TOKEN'])