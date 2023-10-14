import logging
import io
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.model_selection as ms

import azure.functions as func
from azure.storage.blob import BlobServiceClient


def read_csv_to_dataframe(container_client, filename, file_delimiter=','):
    # Digging. Establish a connection to the blob file.
    blob_client = container_client.get_blob_client(blob=filename)

    # Download the blob file.
    blob_download = blob_client.download_blob()

    # Read the blob file into a pandas data frame.
    blob_data = io.StringIO(blob_download.content_as_text())
    dataframe = pd.read_csv(blob_data, delimiter=file_delimiter)

    return dataframe


def get_dataframe(bs_acct_name, bs_container_name, credential, file_name):
    try:
        # Well, would you look at that? The blob storage account is accessed through a webpage.
        bs_acct_url=f'https://{bs_acct_name}.blob.core.windows.net/'

        # Connect to the Azure Blob Storage Account. From there, we will dig down until we get to
        # the file.
        bs_service_client = BlobServiceClient(
            account_url = bs_acct_url,
            credential=credential
        )

        # Digging. This retrieves the container.
        bs_container_client = bs_service_client.get_container_client(container=bs_container_name)

        # Get the dataframe! You'll see the following function digs as well.
        pandas_df = read_csv_to_dataframe(container_client=bs_container_client, filename=file_name)
        # logging.info(pandas_df)

    except Exception as e:
        logging.info(e)
        return (False, None, func.HttpResponse(f"The HTTP triggered function executed unsuccessfully. \n\t {e}!!",
                                    status_code=200))

    return (True, pandas_df, func.HttpResponse("The HTTP triggered function executed successfully."))


# The data and the concepts of the following are from: https://www.kaggle.com/code/ankitakumar/linear-regression-using-wine-quality-dataset
def predict(wines_df):
    # Perform a cursory inspection of the data.
    logging.info(wines_df.head())
    logging.info(wines_df.describe())

    predictors_df = wines_df.drop('quality', axis=1)
    response_df = wines_df['quality']

    # Split the data into training and testing sets.
    predictors_training_df, predictors_testing_df, \
        response_training_df, response_testing_df \
            = ms.train_test_split(predictors_df, response_df, test_size=0.2)

    # Train and predict.
    algorithm = lm.LinearRegression()
    model = algorithm.fit(predictors_training_df, response_training_df)
    prediction = model.predict(predictors_testing_df)

    # Calculate some characteristics of the residuals.
    residuals = response_testing_df.values - prediction

    # Calculate r-squared.
    r_squared = algorithm.score(predictors_training_df, response_training_df)

    results_str = f"The r-squared value of the data is {r_squared}. The mean of the residuals are {np.abs(residuals).mean()} with a standard deviation of {np.abs(residuals).std()}. Overall, linear regression does a fair job predicting the quality of wine."
    logging.info(results_str)
    return func.HttpResponse(results_str)


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # "bs" stands for "blob storage".
    bs_acct_name='winesstore<your id here>'
    bs_container_name='wines-container'
    filename = 'winequality-red.csv'
    credential=<Put your blob storage access key here.>

    (success, wines_df, response) = get_dataframe(bs_acct_name, bs_container_name, credential, filename)
    if success == False:
        return response
    
    response = predict(wines_df)

    return response
