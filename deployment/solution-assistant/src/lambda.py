#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# or in the "license" file accompanying this file. This file is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
# express or implied. See the License for the specific language governing 
# permissions and limitations under the License.

import boto3
import sys

sys.path.append('./site-packages')

from crhelper import CfnResource

helper = CfnResource()

@helper.update
@helper.create
def empty_function(event, _):
    pass

def delete_bucket(event):
    s3_resource = boto3.resource("s3")
    bucket_name = event["ResourceProperties"]["S3BucketName"]
    try:
        s3_resource.Bucket(bucket_name).objects.all().delete()
        print("Successfully deleted objects in bucket "
                "called '{}'".format(bucket_name))
            
    except s3_resource.meta.client.exceptions.NoSuchBucket:
        print(
            "Could not find bucket called '{}'. "
            "Skipping delete.".format(bucket_name)
        )

def delete_sagemaker_endpoint(event):
    sagemaker_client = boto3.client("sagemaker")
    endpoint_name = event["ResourceProperties"]["SageMakerEndpointName"]
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(
            "Successfully deleted endpoint "
            "called '{}'.".format(endpoint_name)
        )
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find endpoint" in str(e):
            print(
                "Could not find endpoint called '{}'. "
                "Skipping delete.".format(endpoint_name)
            )
        else:
            raise e


def delete_sagemaker_endpoint_config(event):
    sagemaker_client = boto3.client("sagemaker")
    endpoint_config_name = event["ResourceProperties"]["SageMakerEndpointConfigName"]

    try:
        sagemaker_client.delete_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        print(
            "Successfully deleted endpoint configuration "
            "called '{}'.".format(endpoint_config_name)
        )
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find endpoint configuration" in str(e):
            print(
                "Could not find endpoint configuration called '{}'. "
                "Skipping delete.".format(endpoint_config_name)
            )
        else:
            raise e


def delete_sagemaker_model(event):
    sagemaker_client = boto3.client("sagemaker")
    model_name = event["ResourceProperties"]["SageMakerModelName"]
    try:
        sagemaker_client.delete_model(ModelName=model_name)
        print("Successfully deleted model called '{}'.".format(model_name))
    except sagemaker_client.exceptions.ClientError as e:
        if "Could not find model" in str(e):
            print(
                "Could not find model called '{}'. "
                "Skipping delete.".format(model_name)
            )
        else:
            raise e

def delete_sagemaker_workteam(event):
    sagemaker_client = boto3.client("sagemaker")
    workteam_name = event["ResourceProperties"]["SolutionPrefix"] + "-workteam"
    try:
        sagemaker_client.delete_workteam(WorkteamName=workteam_name)
        print("Successfully deleted workteam called '{}'.".format(workteam_name))
    except sagemaker_client.exceptions.ClientError as e:
        print(
            "Could not find workteam called '{}'. "
            "Skipping delete.".format(workteam_name)
        )

def delete_sagemaker_workforce(event):
    sagemaker_client = boto3.client("sagemaker")
    workforce_name = "default"
    try:
        sagemaker_client.delete_workforce(WorkforceName=workforce_name)
        print("Successfully deleted workforce called '{}'.".format(workforce_name))
    except sagemaker_client.exceptions.ClientError as e:
        print(
            "Could not find workforce called '{}'. "
            "Skipping delete.".format(workforce_name)
        )

@helper.delete
def on_delete(event, _):
    delete_bucket(event)
    delete_sagemaker_endpoint(event)
    delete_sagemaker_endpoint_config(event)
    delete_sagemaker_model(event)
    delete_sagemaker_workteam(event)
    delete_sagemaker_workforce(event)

def handler(event, context):
    helper(event, context)
