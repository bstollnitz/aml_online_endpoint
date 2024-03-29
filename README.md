# How to deploy using a managed online endpoint and MLflow

This project shows how to deploy a Fashion MNIST MLflow model using an online managed endpoint. Endpoint 1 demonstrates the simplest scenario, endpoint 2 shows how to wrap the deployment with custom code, endpoint 3 exemplifies "aml_token" authentication, and endpoint 4 contains an example of blue-green deployments.

## Blog post

To learn more about the code in this repo, check out the accompanying blog post: https://bea.stollnitz.com/blog/aml-online-endpoint/

## Setup

- You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free) to try it out.
- Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal).
- Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
- Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).
- Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli).
- Install and activate the conda environment by executing the following commands:

```
conda env create -f environment.yml
conda activate aml_online_endpoint
```

- Within VS Code, go to the Command Palette clicking "Ctrl + Shift + P," type "Python: Select Interpreter," and select the environment that matches the name of this project.
- In a terminal window, log in to Azure by executing `az login --use-device-code`.
- Set your default subscription by executing `az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"`. You can verify your default subscription by executing `az account show`, or by looking at `~/.azure/azureProfile.json`.
- Set your default resource group and workspace by executing `az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE>"`. You can verify your defaults by executing `az configure --list-defaults` or by looking at `~/.azure/config`.
- You can now open the [Azure Machine Learning studio](https://ml.azure.com/), where you'll be able to see and manage all the machine learning resources we'll be creating.
- Install the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai), and log in to it by clicking on "Azure" in the left-hand menu, and then clicking on "Sign in to Azure."

## Training and inference on your development machine

- Under "Run and Debug" on VS Code's left navigation, choose the "Train endpoint 1 locally" run configuration and press F5. An 'endpoint_1/model' folder is created with the trained model.
- Repeat for all other endpoints.
- You can analyze the metrics logged in the "mlruns" directory with the following command:

```
mlflow ui
```

- Make a local prediction using the trained mlflow model. You can use either csv or json files:

```
cd aml_online_endpoint/endpoint_1
mlflow models predict --model-uri "model" --input-path "../test_data/images.csv" --content-type csv --env-manager local
mlflow models predict --model-uri "model" --input-path "../test_data/images.json" --content-type json --env-manager local
```

This same syntax should work for all endpoint with the exception of endpoint 2. Here are the commands for endpoint 2:

```
cd aml_online_endpoint/endpoint_2
mlflow models predict --model-uri "pyfunc_model" --input-path "../test_data/images.csv" --content-type csv --env-manager local
mlflow models predict --model-uri "pyfunc_model" --input-path "../test_data/images.json" --content-type json --env-manager local
```

## Deploying in the cloud

## Endpoint 1

```
cd endpoint_1
```

Create the model resource on Azure ML.

```
az ml model create --path model/ --name model-online-1 --version 1 --type mlflow_model
```

Create the endpoint.

```
az ml online-endpoint create -f cloud/endpoint.yml
az ml online-deployment create -f cloud/deployment.yml --all-traffic
```

Invoke the endpoint.

```
az ml online-endpoint invoke --name endpoint-online-1 --request-file ../test_data/images_azureml.json
```

Clean up the endpoint, to avoid getting charged.

```
az ml online-endpoint delete --name endpoint-online-1 -y
```

## Endpoint 2

```
cd ../endpoint_2
```

Create the model resource on Azure ML.

```
az ml model create --path pyfunc_model/ --name model-online-2 --version 1 --type mlflow_model
```

Create the endpoint.

```
az ml online-endpoint create -f cloud/endpoint.yml
az ml online-deployment create -f cloud/deployment.yml --all-traffic
```

Invoke the endpoint.

```
az ml online-endpoint invoke --name endpoint-online-2 --request-file ../test_data/images_azureml.json
```

Invoke the endpoint using a curl command.

```
chmod +x invoke.sh
./invoke.sh
```

Clean up the endpoint, to avoid getting charged.

```
az ml online-endpoint delete --name endpoint-online-2 -y
```

## Endpoint 3

```
cd endpoint_3
```

Create the model resource on Azure ML.

```
az ml model create --path model/ --name model-online-3 --version 1 --type mlflow_model
```

Create the endpoint.

```
az ml online-endpoint create -f cloud/endpoint.yml
az ml online-deployment create -f cloud/deployment.yml --all-traffic
```

Invoke the endpoint.

```
az ml online-endpoint invoke --name endpoint-online-3 --request-file ../test_data/images_azureml.json
```

Clean up the endpoint, to avoid getting charged.

```
az ml online-endpoint delete --name endpoint-online-3 -y
```

## Endpoint 4

```
cd endpoint_4
```

Create the model resource on Azure ML.

```
az ml model create --path model/ --name model-online-4 --version 1 --type mlflow_model
```

Create the endpoint.

```
az ml online-endpoint create -f cloud/endpoint.yml
az ml online-deployment create -f cloud/deployment-blue.yml --all-traffic
az ml online-deployment create -f cloud/deployment-green.yml
az ml online-endpoint update --name endpoint-online-4 --traffic "blue=90 green=10"
```

Invoke the endpoint.

```
az ml online-endpoint invoke --name endpoint-online-4 --request-file ../test_data/images_azureml.json
```

Clean up the endpoint, to avoid getting charged.

```
az ml online-endpoint delete --name endpoint-online-4 -y
```

## Related resources

- [Azure ML endpoints](https://docs.microsoft.com/en-us/azure/machine-learning/concept-endpoints?WT.mc_id=aiml-44166-bstollnitz)
- [Deploying MLflow models](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models?tabs=fromjob%2Cmir%2Ccli?WT.mc_id=aiml-44166-bstollnitz)
- [Ensuring a safe rollout](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-safely-rollout-managed-endpoints?WT.mc_id=aiml-44166-bstollnitz)
