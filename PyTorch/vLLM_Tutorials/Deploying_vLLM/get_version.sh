cd ./measurement

REPO_NAME=$(basename `git rev-parse --show-toplevel`)
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
COMMIT_ID=$(git rev-parse HEAD)

echo "This Measurement file is created using below Repository" > ./measurement_version.txt
echo "Repository Name: $REPO_NAME" >> ./measurement_version.txt
echo "Branch Name: $BRANCH_NAME" >> ./measurement_version.txt
echo "Commit ID: $COMMIT_ID" >> ./measurement_version.txt