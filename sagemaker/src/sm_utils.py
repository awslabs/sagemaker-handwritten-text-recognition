import boto3
import os
import tarfile
from urllib.parse import urlparse

def get_cognito_configs():
    '''
    Helper function to get a previously defined cognito user pool.
    Returns None if there is no previously defined cognitor user pool
    
    Return:
    ------
    cognito_config: {str: str}
        containing the client id and userpool
    '''
    sm_client = boto3.client('sagemaker')
    workforces = sm_client.list_workforces()
    if len(workforces["Workforces"]) > 0:
        return workforces["Workforces"][0]["CognitoConfig"]
    else:
        return None

def get_signup_domain(workteam_name):
    '''

    Returns:
    -------

    '''
    sm_client = boto3.client('sagemaker')
    workteams = sm_client.list_workteams()

    for workteam in workteams["Workteams"]:
        if workteam["WorkteamName"] == workteam_name:
            subdomain = workteam["SubDomain"]
            return "https://{}/logout".format(subdomain)
    return None

def parse_s3_url(s3_url):
    o = urlparse(s3_url, allow_fragments=False)
    return o.netloc, o.path[1:]

def get_latest_training_job(name_contains):
    sagemaker_client = boto3.client('sagemaker')
    response = sagemaker_client.list_training_jobs(
        NameContains=name_contains,
        StatusEquals='Completed'
    )
    training_jobs = response['TrainingJobSummaries']
    assert len(training_jobs) > 0, "Couldn't find any completed training jobs with '{}' in name.".format(name_contains)
    latest_training_job = training_jobs[0]['TrainingJobName']
    return latest_training_job

def get_model_data(training_job):
    sagemaker_client = boto3.client('sagemaker')
    response = sagemaker_client.describe_training_job(TrainingJobName=training_job)
    assert 'ModelArtifacts' in response, "Couldn't find ModelArtifacts for training job."
    return response['ModelArtifacts']['S3ModelArtifacts']

def parse_model_data(s3_location, save_dir):
    s3_client = boto3.client('s3')
    bucket, key = parse_s3_url(s3_location)
    key = key.replace("model.tar.gz", "output.tar.gz")
    s3_client.download_file(Bucket=bucket, Key=key, Filename=os.path.basename(key))
    tar = tarfile.open(os.path.basename(key))
    tar.extractall(path=save_dir)
    tar.close()