#!/usr/bin/env python
import pandas as pd
import datetime


def compute_indicators(df):
    df_ind = df.copy(True)

    # Bollinger bands
    b_up, b_mid, b_low = bollinger(df_ind.close)
    df_ind.loc[:, "bollinger_upper"] = b_up
    df_ind.loc[:, "bollinger_middle"] = b_mid
    df_ind.loc[:, "bollinger_lower"] = b_low

    # RSI
    df_ind.loc[:, "rsi"] = rsi(df_ind.close)

    # Remove the previous period (when the indicators are not representative)
    timestamp = df_ind.index[0] + datetime.timedelta(days=21)
    return df_ind.loc[df_ind.index > timestamp]


def bollinger(series, days=21):
    rolling = series.rolling("{}D".format(days))
    middle_band = rolling.mean()
    std = rolling.std()
    upper_band = middle_band + 1.5 * std
    lower_band = middle_band - 1.5 * std

    # Set the new fields
    return upper_band, middle_band, lower_band


def rsi(series, days=14):
    delta = series.diff().copy()
    window = "{}D".format(days)

    up_diff, down_diff = delta.copy(), delta.copy()
    up_diff[delta <= 0] = 0.
    down_diff[delta > 0] = 0.

    rs_up = up_diff.rolling(window).mean()
    rs_down = down_diff.rolling(window).mean().abs()

    return 100. - 100. / (1. + rs_up / rs_down)


def compute_trend(amount1, amount2, percentage=.01):
    return ((amount2 - amount1) / amount1) > percentage


def to_supervised(df, period):
    seq = range(period, 0, -1)
    values = [df.shift(i) for i in seq]
    sup_df = pd.concat(values, axis=1)
    sup_df.columns = ["{}_{}".format(column, i)
                      for i in seq for column in df.columns]

    # Remove NaNs
    return sup_df.dropna()


def main(input_file, output_file):
    # Load dataframe
    df = pd.read_csv(input_file, skiprows=1)

    # Filter & rename columns
    df.drop("Symbol", axis=1, inplace=True)
    df.columns = ["timestamp", "open", "high", "low", "close", "volume_from",
                  "volume_close"]

    # Format & index timestamp
    df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %I-%p")
    df.set_index("timestamp", inplace=True)

    # Reverse
    df = df.iloc[::-1]

    # Filter by the consecutive rows
    diff = pd.DataFrame(dict(timestamp=df.index))
    diff.loc[:, "delta"] = diff.timestamp.diff()
    df = df[(diff.delta == pd.Timedelta("1 hour")).values]

    # Compute the technical indicators
    df = compute_indicators(df)

    # Create the supervised dataset in a window of 3 hours
    period = 3 # 3 hours
    sup_df = to_supervised(df, period)
    df = df.iloc[period:]

    # Label the dataset with the actual trend
    sup_df["trend"] = compute_trend(sup_df.close_1, df.high)

    # Save to file
    sup_df.to_csv(output_file)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: {} <input_file> <output_file>".format(sys.argv[0]))
        sys.exit(1)

    main(*sys.argv[1:])
