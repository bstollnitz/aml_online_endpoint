## Endpoint 1

```
cd aml-managed-endpoint-mlflow/endpoint-1
```

```
az ml model create --path model/ --name model-managed-mlflow-1 --version 1 
```

```
az ml online-endpoint create -f cloud/endpoint.yml
az ml online-deployment create -f cloud/deployment.yml --all-traffic
```

```
az ml online-endpoint invoke --name endpoint-managed-mlflow-1 --request-file ../test-data/images_azureml.json
```


## Endpoint 2

```
cd aml-managed-endpoint-mlflow/endpoint-2
```

```
az ml model create --path model/ --name model-managed-mlflow-2 --version 1 
```

```
az ml online-endpoint create -f cloud/endpoint.yml
az ml online-deployment create -f cloud/deployment.yml --all-traffic
```

```
az ml online-endpoint invoke --name endpoint-managed-mlflow-2 --request-file ../test-data/images_azureml.json
```
