import json
import boto3
import re

def split_s3_path(s3_uri):
    '''
    Helper function to split an s3 uri (s3://<bucket>/<key>) 
    to `bucket` and `key`.
    '''
    bucket = s3_uri[5:].split("/")[0]
    key = "/".join(s3_uri[5:].split("/")[1:])
    return bucket, key
    
def parse_label(label):
    '''
    Helper function to parse the label in the form of 
    "Text: <text>, Line #: <num>, Type: <text ftype>"
    '''
    text = re.search(r"Text:([A-Za-z0-9\s]+),", label).group(1)
    text = text.replace(" ", "")
    
    line_num = re.search(r"Line #:([0-9\s]+),", label).group(1)
    line_num = line_num.replace(" ", "")
    
    word_type = re.search(r"Type:([A-Za-z\s]+)", label).group(1)
    word_type = word_type.replace(" ", "")
    
    return text, line_num, word_type 

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    labeling_job_arn = event["labelingJobArn"]
    label_attribute_name = event["labelAttributeName"]

    consolidated_labels = []

    s3_uri = event['payload']['s3Uri']
    bucket, key = split_s3_path(s3_uri)
    
    s3 = boto3.client('s3')
    textFile = s3.get_object(Bucket=bucket, Key=key)
    filecont = textFile['Body'].read()
    annotations = json.loads(filecont)
    
    for dataset in annotations:
        for annotation in dataset['annotations']:
            new_annotation = json.loads(annotation['annotationData']['content'])
            
            texts = []
            for label in new_annotation['transcription']['polygons']:
                text, line_num, word_type = parse_label(label["label"])
                vertices = label["vertices"]
                texts.append({
                    "text": text,
                    "line_num": line_num,
                    "type": word_type,
                    "bb": vertices
                })

            label = {
                'datasetObjectId': dataset['datasetObjectId'],
                'consolidatedAnnotation' : {
                    'content': {
                        label_attribute_name: {
                            'texts': texts,
                            'imageSource': dataset['dataObject']
                            }
                        }
                    }
                }
            consolidated_labels.append(label)
    print("Consolidated labels \n {}".format(consolidated_labels))
    return consolidated_labels
