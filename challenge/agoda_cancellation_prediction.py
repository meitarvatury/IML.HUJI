from sklearn.base import BaseEstimator

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd

LABEL = "cancellation_datetime"
DATES = ["booking_datetime", "checkin_date", "checkout_date"]
DUMMIES = ["hotel_id", "hotel_country_code", "origin_country_code", "accommadation_type_name", "charge_option"]
DROP = ["h_booking_id", "hotel_live_date", "h_customer_id", "guest_is_not_the_customer", "customer_nationality",
           "guest_nationality_country_name", "no_of_adults", "no_of_children", "no_of_extra_bed",
           "no_of_room", "language", "original_payment_method", "original_payment_type",
           "original_payment_currency", "is_user_logged_in", "is_first_booking",
           "request_nonesmoke", "request_highfloor", "request_largebed", "request_twinbeds",
           "hotel_area_code", "hotel_brand_code", "hotel_chain_code", "hotel_city_code"]

def cancellacion(c_policy, days_before_cancel, price_per_night, price_total, no_show=False):
    if (c_policy == "UNKNOWN"):
        return 0
    if no_show:
        days_before_cancel = 0
    i = 0
    days = 0
    payment = 0
    last_payment = 0
    ns = 0
    while (c_policy[i:]):
        temp = 0
        while (c_policy[i].isdigit()):
            temp = temp * 10 + int(c_policy[i])
            i += 1
        if (c_policy[i] == "D"):
            ns += 1
            days = temp
            i += 1
            if (days_before_cancel > days):
                return payment
        elif (c_policy[i] == "N"):
            ns -= 1
            last_payment = payment
            payment = temp * price_per_night
            i += 1
        elif (c_policy[i] == "P"):
            ns -= 1
            last_payment = payment
            payment = (temp / 100) * price_total
            i += 1
        if (len(c_policy)>i):
            if (c_policy[i] == "_"):
                i += 1  # the tap "_"
        else:
            break
    if no_show:
        return payment
    if ns:
        return last_payment
    return payment


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename, parse_dates=DATES)
    full_data.fillna(0, inplace=True)
    dummies = pd.get_dummies(data=full_data, columns=DUMMIES)
    stay_dates = (full_data["checkout_date"] - full_data["checkin_date"])
    time_to_stay = (full_data["checkin_date"] - full_data["booking_datetime"])
    labels = full_data[LABEL]

    full_data.drop(columns=DROP + DUMMIES + DATES + [LABEL], inplace=True)
    full_data = pd.concat([full_data, dummies, time_to_stay, stay_dates], axis=1)

    return full_data, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    #train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)

    # Fit model over data
    #estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    p_loss_mean = []
    for i in range(20):
        train_X, train_y, test_X, test_y = split_train_test(df, cancellation_labels)
        estimator = AgodaCancellationEstimator().fit(train_X, train_y)
        p_loss_mean.append(estimator.loss(test_X, test_y))

    p_loss_mean = np.array(p_loss_mean)
    p_loss_mean.mean()

    # Store model predictions over test set
    evaluate_and_export(estimator, test_X, "313597122_316419423_206990418.csv")

