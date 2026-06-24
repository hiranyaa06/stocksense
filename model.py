import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data import get_stock_data

def create_features(df): #(feature enginnering)
    
    df = df.copy()

    # prices from past days
    for i in range(1, 6):
        df[f'lag_{i}'] = df['Close'].shift(i)

    # Rolling mean features — average of last 3 and 5 days- done to smooth out noise as stock prices are often fluctuating
    df['rolling_3'] = df['Close'].rolling(3).mean()
    df['rolling_5'] = df['Close'].rolling(5).mean()
    df['rolling_10'] = df['Close'].rolling(10).mean()
    df['rolling_20'] = df['Close'].rolling(20).mean()

    # Time based features
    df['day']   = df.index.day
    df['month'] = df.index.month
    df['year']  = df.index.year

    # Target:tomorrow's price
    df['tomorrow'] = df['Close'].shift(-1)

    # Drop NaN rows
    df = df.dropna()

    return df


def train_model(ticker):
    """
    Fetches data, creates features, trains Random Forest model:
    """

    # Get real stock data
    raw_df = get_stock_data(ticker)

    # Create features
    df = create_features(raw_df)

    # Separate X and y
    feature_cols = [f'lag_{i}' for i in range(1, 6)] + \
               ['rolling_3', 'rolling_5',
                'rolling_10', 'rolling_20',
                'day', 'month', 'year']

    X = df[feature_cols]
    y = df['tomorrow']

    # Time based split (80% train, 20% test):
    #(it cant be a random split cz time series analysis is BASED on CHRONOLOGICAL ordering of the datasets)
    split = int(len(df) * 0.8)
    X_train, y_train = X.iloc[:split], y.iloc[:split] #frm 0m to 80% of the total amt of data
    X_test,  y_test  = X.iloc[split:], y.iloc[split:] #the last remaining 20% of the data

    # Train Random Forest!
    model = RandomForestRegressor(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    mae  = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"Model has been trained :) ")
    print(f"   MAE  : ${mae:.2f}")
    print(f"   RMSE : ${rmse:.2f}")

    return model, df, feature_cols, mae, rmse


def predict_7_days(model, df, feature_cols):
    """
    Predicts next 7 days recursively, aka using recursive forecasting
    JUST AN ADDITION FEATURE FOR THE APP LOL
    """

    # Convert to plain Python floats- to prevent typw mismatch to pandas (btwn py float to numpy.float64)
    last_prices = [float(x) for x in df['Close'].values[-20:]]
    predictions = []

    for day in range(7):

        lags = [
            last_prices[-1],
            last_prices[-2],
            last_prices[-3],
            last_prices[-4],
            last_prices[-5]
        ]

        rolling_3 = float(np.mean(last_prices[-3:]))
        rolling_5 = float(np.mean(last_prices[-5:]))
        rolling_10 = float(np.mean(last_prices[-10:]))
        rolling_20 = float(np.mean(last_prices[-20:]))

        next_date = df.index[-1] + pd.Timedelta(days=day + 1)

        X_future = pd.DataFrame([[
            *lags,
            rolling_3,
            rolling_5,
            rolling_10,
            rolling_20,
            next_date.day,
            next_date.month,
            next_date.year
        ]], columns=feature_cols)

        next_price = float(model.predict(X_future)[0])

        predictions.append({
            'date': next_date,
            'price': round(next_price, 2)
        })

        last_prices.append(next_price)

    return pd.DataFrame(predictions).set_index('date')


# testing
if __name__ == "__main__":
    model, df, feature_cols, mae, rmse = train_model("AAPL")
    preds = predict_7_days(model, df, feature_cols)

    print("\n Next 7 days:")
    print(preds)