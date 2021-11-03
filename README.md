# Patent-Classifier


## Setup virtual environment

Activate virtual environment.

```bash
python3 -m venv venv
. venv/bin/activate
```

Using Anaconda in MacOS/Linux.

```bash
conda create --name venv python=3.8 numpy
source activate venv
```


Using Anaconda in Windows.

```bash
conda create --name venv python=3.8 numpy scipy
activate venv
```

Install dependencies.

```bash
pip install -r requirements.txt
```


## Setup Google cloud
From https://github.com/tensorflow/cloud.

1.  Install [gcloud sdk](https://cloud.google.com/sdk/docs/install) and verify that it is installed.

    ```shell
    which gcloud
    ```

1. Install newer version of Tensorflow cloud.

    ```shell
    git clone https://github.com/tensorflow/cloud.git
    cd cloud
    pip install src/python/.
    ```

1.  Set default gcloud project.

    ```shell
    export PROJECT_ID=<your-project-id>
    gcloud config set project $PROJECT_ID
    ```

1.  Create a service account.

    ```shell
    export SA_NAME=<your-sa-name>
    gcloud iam service-accounts create $SA_NAME
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member serviceAccount:$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com \
        --role 'roles/editor'
    ```

1.  Create a key for your service account.

    ```shell
    gcloud iam service-accounts keys create ~/key.json --iam-account $SA_NAME@$PROJECT_ID.iam.gserviceaccount.com
    ```

1.  Create the GOOGLE_APPLICATION_CREDENTIALS environment variable.

    ```shell
    export GOOGLE_APPLICATION_CREDENTIALS=~/key.json
    ```



## Running


### Train locally

This can be used to train a model by using an experiment config. For example:
```bash
python3 train.py --flagfile=experiments/<experiment config>
```

If you want you can specify the flags by using command line args. For example:
```bash
python3 train.py --num_epochs=100
```


### Train on Google Cloud

Use the flag use_cloud to train the model on Google Cloud. The flag machine_config can be used to change the type of GPU.
```bash
python3 train.py --use_cloud
```
