# Handwriting Recognition using Amazon SageMaker

The SageMaker handwriting recognition solution applies deep learning techniques to transcribe text in images of passages into strings.
If you have your own data, you can use this solution to label your own data and train a new network with it. Endpoints are then automatically deployed with the solution.

![](https://sagemaker-solutions-prod-us-east-2.s3.us-east-2.amazonaws.com/sagemaker-handwriting-recognition/docs/htr_demo.png)

## Getting Started with Amazon SageMaker

You will need an AWS account to use this solution. Sign up for an account [here](https://aws.amazon.com/).

To run this JumpStart 1P Solution and have the infrastructure deploy to your AWS account you will need to create an active SageMaker Studio instance (see Onboard to Amazon SageMaker Studio). When your Studio instance is Ready, use the instructions in SageMaker JumpStart to 1-Click Launch the solution.

The solution artifacts are included in this GitHub repository for reference.

*Note*: Solutions are available in most regions including us-west-2, and us-east-1.

**Caution**: Cloning this GitHub repository and running the code manually could lead to unexpected issues! Use the AWS CloudFormation template. You'll get an Amazon SageMaker Notebook instance that's been correctly setup and configured to access the other resources in the solution.

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
