# How to deploy using a managed online endpoint and MLflow

This project shows how to deploy a Fashion MNIST MLflow model using an online managed endpoint. Endpoint 1 demonstrates the simplest scenario, endpoint 2 shows how to wrap the deployment with custom code, endpoint 3 exemplifies "aml_token" authentication, and endpoint 4 contains an example of blue-green deployments.

## Azure setup

* You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free?WT.mc_id=aiml-44166-bstollnitz) to try it out.
* Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal?WT.mc_id=aiml-44166-bstollnitz).
* Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?WT.mc_id=aiml-44166-bstollnitz). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
* If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace."
* Alternatively, if you plan to use your local machine:
  * Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?WT.mc_id=aiml-44166-bstollnitz).
  * Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?WT.mc_id=aiml-44166-bstollnitz).
* In a terminal window, login to Azure by executing `az login --use-device-code`. 
* Set your default subscription by executing `az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"`. You can verify your default subscription by executing `az account show`, or by looking at `~/.azure/azureProfile.json`.
* Set your default resource group and workspace by executing `az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE>"`. You can verify your defaults by executing `az configure --list-defaults` or by looking at `~/.azure/config`.
* You can now open the [Azure Machine Learning studio](https://ml.azure.com/?WT.mc_id=aiml-44166-bstollnitz), where you'll be able to see and manage all the machine learning resources we'll be creating.
* Although not essential to run the code in this post, I highly recommend installing the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai).



## Project setup

If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace."

Alternatively, you can set up your local machine using the following steps.

Install conda environment:

```
conda env create -f environment.yml
```

Activate conda environment:

```
conda activate aml_online_endpoint
```


## Training and inference on your development machine

* Open the 'endpoint_1/src/train.py` file and press F5. An 'endpoint_1/model' folder is created with the trained model. Repeat for all other endpoints.
* You can analyze the metrics logged in the "mlruns" directory with the following command:

```
mlflow ui
```

* Make a local prediction using the trained mlflow model. You can use either csv or json files:

```
cd aml_online_endpoint/endpoint_1
mlflow models predict --model-uri model --input-path "../test_data/images.csv" --content-type csv
mlflow models predict --model-uri model --input-path "../test_data/images.json" --content-type json
```

This same syntax should work for all endpoint with the exception of endpoint 2. Here are the commands for endpoint 2:

```
cd aml_online_endpoint/endpoint_2
mlflow models predict --model-uri pyfunc_model --input-path "../test_data/images.csv" --content-type csv
mlflow models predict --model-uri pyfunc_model --input-path "../test_data/images.json" --content-type json
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
