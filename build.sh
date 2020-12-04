#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Please provide the solution name as well as the base S3 bucket name and the region to run build script."
    echo "For example: ./build.sh trademarked-solution-name sagemaker-solutions-build us-west-2"
    exit 1
fi

mkdir build

# Package the ground truth lambdas
mkdir build/ground-truth

mkdir build/ground-truth/pre-label
cp -r ./sagemaker/ground_truth/pre-label ./build/ground-truth/
find  ./build/ground-truth/pre-label -name '*.pyc' -delete
(cd ./build/ground-truth/pre-label && zip ../../pre-label-gt.zip *)

mkdir build/ground-truth/post-label
cp -r ./sagemaker/ground_truth/post-label ./build/ground-truth/
find  ./build/ground-truth/post-label -name '*.pyc' -delete
(cd ./build/ground-truth/post-label && zip -q -r9 ../../post-label-gt.zip *)

rm -rf ./build/ground-truth

# Package the solution assistant
mkdir build/solution-assistant
cp -r ./deployment/solution-assistant ./build/
(cd ./build/solution-assistant && pip install -r requirements.txt -t ./src/site-packages)
find ./build/solution-assistant -name '*.pyc' -delete
(cd ./build/solution-assistant/src && zip -q -r9 ../../solution-assistant.zip *)
rm -rf ./build/solution-assistant

# Package the string functions
mkdir build/string-functions
cp -r ./deployment/string-functions ./build/
(cd ./build/string-functions && pip install -r requirements.txt -t ./src/site-packages)
find ./build/string-functions -name '*.pyc' -delete
(cd ./build/string-functions/src && zip -q -r9 ../../string-functions.zip *)
rm -rf ./build/string-functions

# # Upload to S3
s3_prefix="s3://$2-$3/$1"
echo "Using S3 path: $s3_prefix"

# Copy training assets
aws s3 cp --recursive sagemaker $s3_prefix/sagemaker --exclude '.*' --exclude "*~" --exclude "credentials.json" --exclude "Config"

aws s3 cp --recursive deployment $s3_prefix/deployment --exclude '.*' --exclude "*~"
aws s3 cp --recursive docs $s3_prefix/docs --exclude '.*' --exclude "*~"

aws s3 cp --recursive build $s3_prefix/build
aws s3 cp Readme.md $s3_prefix/

# Copy solution artefacts to the folder
aws s3 cp "s3://sagemaker-solutions-artifacts/$1/model-csafe.tar.gz" $s3_prefix/build/model.tar.gz
