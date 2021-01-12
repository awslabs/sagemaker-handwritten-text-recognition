# Handwriting Recognition using Amazon SageMaker

The SageMaker handwriting recognition solution applies deep learning techniques to transcribe text in images of passages into strings.
If you have your own data, you can use this solution to label your own data and train a new network with it. Endpoints are then automatically deployed with the solution.

![](https://sagemaker-solutions-prod-us-east-2.s3.us-east-2.amazonaws.com/sagemaker-handwriting-recognition/docs/htr_demo.png)

## Getting Started with Amazon SageMaker

### Get an AWS account

You will need an AWS account to use this solution. Sign up for an account here (https://aws.amazon.com/).
You will also need to have permission to use AWS CloudFormation (https://aws.amazon.com/cloudformation/) and to create all the resources detailed in the architecture section. All AWS permissions can be managed through AWS IAM (https://aws.amazon.com/iam/). Admin users will have the required permissions, but please contact your account's AWS administrator if your user account doesn't have the required permissions.

#### Architecture

![](https://sagemaker-solutions-prod-us-east-2.s3.us-east-2.amazonaws.com/sagemaker-handwriting-recognition/docs/archi.png)

The following services are used:
- Amazon S3: To store datasets, training job information, labelling information, and network artifacts
- Amazon SageMaker Notebooks: Used to preprocess and visualise the data, and to train the deep learning models
- Amazon SageMaker Ground Truth: Used to label your custom data
- Amazon SageMaker Endpoint: Used to deploy the trained model

### Cost

You will be given credits to use any AWS service, please contact AICrowd for details. 

You are responsible for the cost of the AWS services used while running this solution. For details refer to the pricing listed at [Amazon SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/).

As of September 1, 2020, the Amazon SageMaker training cost (excluding notebook instance) are listed as:

* ml.p2.xlarge	
* ml.g4dn.4xlarge	$1.686 per hour (1 GPU, 16 vCPU)
* ml.p3.2xlarge	$4.284 per hour (1 GPU, 8 vCPU)

### Launch the solution

While logged on to your AWS account, click on the link to quick create the AWS CloudFormation Stack for the region you want to run your notebook:

<table>
  <tr>
    <th colspan="3">AWS Region</td>
    <th>AWS CloudFormation</td>
  </tr>
  <tr>
    <td>US West</td>
    <td>Oregon</td>
    <td>us-west-2</td>
    <td align="center">
      <a href="https://us-west-2.console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/create/review?templateURL=https://sagemaker-solutions-prod-us-west-2.s3-us-west-2.amazonaws.com/sagemaker-handwriting-recognition/deployment/template.yaml&stackName=sagemaker-soln-htr&param_SolutionPrefix=sagemaker-soln-htr-handwriting&param_S3BucketName=bucket">
        <img src="docs/launch_button.svg" height="30">
      </a>
    </td>
  </tr>
</table>


Enter your desired bucket name is in the **SageMaker Configurations section**.

### Stages

The solution is split into the following stages. Each stage has it's own notebook
- Demo: You can try a demo SageMaker endpoint with an image of a handwritting passage
- Introduction (here): a high-level overview of the solution
- Label own data: notebook to prepare your own dataset for labelling
- Visualise your own data
- Model training: notebook to train models with your labelled dataset
- Endpoint updates: Notebooks to build SageMaker endpoints with the trained mode

## License

This project is licensed under the Apache-2.0 License.
