import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df.fillna(0, inplace=True)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df = df.drop(df[df['Temp'] < -40].index)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("C:/Users/meitar vatury/OneDrive/Desktop/school/secondyear/IML/IML.HUJI/datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    X_israel = X[X['Country'] == "Israel"]
    fig1 = px.scatter(X_israel, x="DayOfYear", y="Temp", color=X_israel["Year"].astype(str),
                      title="The average temperatures by day in Israel between the years 1995-2007")

    fig1.show()

    X_by_months = X_israel.groupby(['Month'])
    month_std = X_by_months.agg('std')
    fig2 = px.bar(month_std, y="Temp")
    fig2.update_layout(title=" standard deviation of the daily temperatures by months",
                       xaxis_title="Month",
                       yaxis_title="standard deviation of the temperatures")

    fig2.show()

    # Question 3 - Exploring differences between countries
    X_country_month = X.groupby(['Country', 'Month'])
    month_mean = X_country_month.agg('mean')
    month_std = X_country_month.agg('std')

    fig3 = px.line(month_mean, x=month_mean.axes[0].get_level_values(1), y='Temp',
                   color=month_mean.axes[0].get_level_values(0), error_y=month_std['Temp'])

    fig3.update_layout(title="average monthly temperature by country",
                       xaxis_title="months",
                       yaxis_title="temperature")

    fig3.show()

    # Question 4 - Fitting model for different values of `k`
    y_israel = pd.Series(X_israel['Temp'])
    loss_by_k = []
    training_set, training_response, test_set, test_response = split_train_test(X_israel, y_israel, 0.75)
    for k in range(1, 11):
        polynomial_model = PolynomialFitting(k).fit(training_set['DayOfYear'].to_numpy(), training_response.to_numpy())
        model_loss = np.round(polynomial_model.loss(test_set['DayOfYear'], test_response), 2)
        loss_by_k.append(model_loss)
        print(model_loss)
    fig4 = px.bar(x=range(1, 11), y=loss_by_k, text_auto=True)
    fig4.update_layout(title="The estimated error of the model as a function of k",
                       xaxis_title="k- degree of the polynomial model",
                       yaxis_title="MSE loss")
    fig4.show()


    # Question 5 - Evaluating fitted model on different countries
    polynomial_model_5 = PolynomialFitting(5).fit(X_israel['DayOfYear'], X_israel['Temp'])
    countries = ["South Africa", "The Netherlands", "Jordan"]
    countries_loss = []
    for country in countries:
        x_country = X[X['Country'] == country]
        model_loss = polynomial_model_5.loss(x_country['DayOfYear'], x_country['Temp'])
        countries_loss.append(model_loss)

    fig5 = px.bar(x=countries, y=countries_loss)
    fig5.update_layout(title="error by country from the model fitted to Israel",
                       xaxis_title="countries",
                       yaxis_title="MSE loss")
    fig5.show()