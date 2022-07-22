# Created by: Jess Gallo
# Date Created: 7/21/2022
# Last Modified: 7/22/2022
# Description: Forage - Cognizant Virtual Experience - Task 4 - Python Module


# Libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Global constants
SPLIT = 75
k = 10


# Load data
def load_data(path: str = "/path/to/csv/"):
    """
    This function takes a path string to a CSV file and loads it into a DataFrame
    :param path: path (optional): str, relative path of the CSV file
    :return: df: pd.DataFrame
    """

    # df = pd.read_csv(f'{path}', index_col=0)
    df = pd.read_csv(f'{path}', index_col=0)
    # df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
    return df


# Create a target variable and predictor variables
def create_target_and_predictors(data: pd.DataFrame = None, target: str = 'estimated_stock_pct'):
    """
    This function takes in a DataFrame and splits the columns into a target column and a set of
    predictor variables, i.e. x & y. These 2 splits of the data will be used to train a supervised
    machine learning model.
    :param data: pd.DataFrame, dataframe containing data for the model
    :param target: str(optional), target variable that you want to predict
    :return: x: pd.DataFrame
             y: pd.Series
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f'Target:  {target} is not present in the data')

    x = data.drop(columns=[target])
    y = data[target]
    return x, y


# Train algorithm
def train_algo_with_cross_val(x: pd.DataFrame = None, y: pd.Series = None):
    """
    This function takes the predictor and target variables and trains a Random
    Forest Regressor model across K folds. Using cross-validation, performance
    metrics will be output for each fold during training.
    :param x: pd.DataFrame, predictor variables
    :param y: pd.Series, target variable
    :return:
    """

    # Create a list that will store the accuracies of each fold
    accuracy = []

    # Enter a loop to run K folds of cross-validation
    for fold in range(0, k):
        # Instantiate algorithm and scaler
        model = RandomForestRegressor()
        scaler = StandardScaler()

        # Create training and test splits
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=SPLIT, random_state=42)

        # Scale x data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        # Train model
        trained_model = model.fit(x_train, y_train)

        # Generate predictions on test samples
        y_pred = trained_model.predict(x_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        accuracy.append(mae)
        print(f'Fold {fold + 1}: MAE = {mae:.3f}')

    # Finish by computing the average MAE across all folds
    print(f'Average MAW: {(sum(accuracy) / len(accuracy)):.2f}')


if __name__ == '__main__':
    df = load_data("C://Users//Gallo//Jupyter//merged_sales_data.csv")
    x, y = create_target_and_predictors(df)
    train_algo_with_cross_val(x, y)
