from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import math
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df.fillna(0, inplace=True)

    df = df.drop(df[df['price'] == 0].index)
    df = df.drop(df[df['price'] == 'nan'].index)
    df = df.drop(df[df['sqft_living'] < df['bedrooms'] * 150].index)

    price_y = pd.Series(df['price'])
    price_y[price_y < 0] = -price_y

    df = pd.get_dummies(df, columns=['zipcode'], dummy_na=True)
    df.drop(columns=['id', 'date', 'price'], inplace=True)   # remove irrelevant values
    df.drop('sqft_living', axis=1, inplace=True)   # Linear dependent in sqft_above and sqft_basement
    df.drop(columns=['long'], inplace=True)    # not enough information about the location of the house

    np.where(df < 0, 0, df)

    df = df.drop(df[df['floors'] == 0].index)
    df['yr_renovated'] = df[["yr_renovated", "yr_built"]].max(axis=1)

    return df, price_y


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for (columnName, x) in X.iteritems():
        if np.sum((x-x.mean()) ** 2) != 0:
            pearson_corr = (x - x.mean()).dot(y - y.mean()) / \
                           (math.sqrt(np.sum((x-x.mean()) ** 2)) * math.sqrt(np.sum((y-y.mean()) ** 2)))
            fig1 = go.Figure(data=go.Scatter(x=x, y=y, mode='markers'))

            fig1.update_layout(title=columnName+" pearson_correlation = " + str(pearson_corr),
                               xaxis_title=columnName,
                               yaxis_title="the response")

            fig1.write_image(output_path+"/"+columnName+".png")




if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, y = load_data("C:/Users/meitar vatury/OneDrive/Desktop/school/secondyear/IML/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, y, "C:/Users/meitar vatury/OneDrive/Desktop/school/secondyear/IML/IML.HUJI/exercises")

    # Question 3 - Split samples into training- and testing sets.
    training_set, training_response, test_set, test_response = split_train_test(df, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    p_range = np.array(range(10, 101))
    p_loss_mean = []
    p_loss_std = []
    for p in p_range:
        cur_p_loss = []
        for i in range(10):
            train_x, train_y, temp_x, temp_y = split_train_test(training_set, training_response, p/100)
            linear_model = LinearRegression().fit(train_x, train_y)
            model_loss = linear_model.loss(test_set, test_response)
            cur_p_loss.append(model_loss)
        cur_p_loss = np.array(cur_p_loss)
        p_loss_mean.append(np.mean(cur_p_loss))
        p_loss_std.append(np.std(cur_p_loss))
    p_loss_mean = np.array(p_loss_mean)
    p_loss_std = np.array(p_loss_std)

    fig1 = go.Figure([
        go.Scatter(
            name='MSE loss',
            x=p_range,
            y=p_loss_mean,
            mode='lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=p_range,
            y=p_loss_mean + (2 * p_loss_std),
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=p_range,
            y=p_loss_mean - (2 * p_loss_std),
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])

    fig1.update_layout(title=" the MSE loss mean as function of the sample size ",
                       xaxis_title="p range (%)",
                       yaxis_title="the loss mean")
    fig1.show()






