import io
import os
import pandas as pd
import logging

import azure.functions as func


def main(smallcsvblob: func.InputStream):
    logging.info(f"Python blob trigger function processed blob \n"
                 f"Name: {smallcsvblob.name}\n"
                 f"Blob Size: {smallcsvblob.length} bytes")
    base_file_name = os.path.basename(smallcsvblob.name)
    if base_file_name == 'small.csv':
        logging.info(f"Handling small.csv...")

        # Read the contents of the blob.
        file_contents = smallcsvblob.read()

        # Convert to a string.
        csv_str = file_contents.decode('utf-8')

        # Create a pandas DataFrame.
        df = pd.read_csv(io.StringIO(csv_str))
        logging.info(df)
