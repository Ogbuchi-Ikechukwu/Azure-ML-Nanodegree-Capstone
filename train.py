import argparse
import joblib
import numpy as np
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from azureml.core import Dataset, Datastore
from azureml.core import Workspace, Experiment
from azureml.core.run import Run
from azureml.data.datapath import DataPath
from azureml.data.dataset_factory import TabularDatasetFactory


def main():
    # Get run context
    run = Run.get_context()
    
    # Parse data
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_url', type=str, default='https://raw.githubusercontent.com/Ogbuchi-Ikechukwu/Azure-ML-Nanodegree-Capstone/master/starter_file/train_data_cleaned.csv', help='URL of the dataset to be used')
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength", np.float(args.C))
    run.log("Max iterations", np.int(args.max_iter))

    # Get the dataset
    dataset = Dataset.Tabular.from_delimited_files(args.data_url)

    # Drop non-numeric columns from the dataset
    df = dataset.to_pandas_dataframe()
    df = df.drop(['Customer Id', 'Building_Painted', 'Building_Fenced','Garden','Settlement'], axis=1)
    
    #a function to help me handle nan or infinity values
    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)
    df=clean_dataset(df)
    # Separate x and y columns. 'Survived' is y, because it is was we want to predict
    y_df = df['Claim']
    df.pop('Claim')
    x_df = df

    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.25)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()