from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import statsmodels.api as sm
import yfinance as yf
import seaborn as sns
import pandas as pd
import numpy as np

def fetch_data(ticker, start_date='2020-01-01', end_date='2024-01-01'):
    return yf.download(ticker, start=start_date, end=end_date)

def ewmac(price, vol, Lfast, Lslow):
    fast_ewma = price.ewm(span=Lfast, min_periods=1).mean()
    slow_ewma = price.ewm(span=Lslow, min_periods=1).mean()
    raw_ewmac = fast_ewma - slow_ewma
    return raw_ewmac / vol.ffill()

def calculate_ewmac_signals(data, parameters):
    composite_signal = pd.Series(0, index=data.index)
    for filter_speed in parameters:
        Lfast = filter_speed
        Lslow = 4 * filter_speed
        signal_name = f'EWMAC_{filter_speed}'
        data[signal_name] = scale_cap_forecast(ewmac(data['Close'], data['Volatility'], Lfast, Lslow))
        composite_signal += data[signal_name]
    data['Composite_Signal'] = composite_signal / len(parameters) 
    return data

def scale_cap_forecast(signals, cap=20):
    signals = signals * 10 / signals.abs().mean()
    return signals.clip(-cap, cap)

def adjust_trade_buffer(data, max_loss_sharpe, window=30):
    rolling_mean = data['Log_Returns'].rolling(window=window).mean()
    rolling_std = data['Log_Returns'].rolling(window=window).std()
    rolling_sharpe = rolling_mean / rolling_std
    trade_buffer = (rolling_sharpe.abs() / max_loss_sharpe).clip(lower=1)
    return trade_buffer

def calculate_pnl(trades, data, initial_capital, transaction_cost=0.001, holding_cost_rate=0.0001):
    trades['Position Change'] = trades['Position'].diff().fillna(0)
    trades['Position Cost'] = 0.0
    trades['Realized PnL'] = 0.0
    trades['Unrealized PnL'] = 0.0
    trades['Total PnL'] = 0.0
    trades['Account Size'] = initial_capital
    trades['Avg Position Price'] = 0.0
    trades['Holding Cost'] = 0.0

    for i in range(1, len(trades)):
        position_change = trades['Position Change'].iloc[i]
        execution_price = data['Close'].iloc[i]
        previous_position = trades['Position'].iloc[i-1]
        previous_position_cost = trades['Position Cost'].iloc[i-1]
        avg_position_price = trades['Avg Position Price'].iloc[i-1]

        # Update position cost and average price per unit
        if position_change != 0:
            if previous_position + position_change != 0:
                new_cost = previous_position_cost + position_change * execution_price
                trades.loc[trades.index[i], 'Position Cost'] = new_cost
                trades.loc[trades.index[i], 'Avg Position Price'] = new_cost / abs(previous_position + position_change)
            else:
                trades.loc[trades.index[i], 'Position Cost'] = 0
                trades.loc[trades.index[i], 'Avg Position Price'] = 0

            # Calculate realized PnL
            if np.sign(position_change) != np.sign(previous_position):
                realized_pnl = (execution_price - avg_position_price) * min(abs(position_change), abs(previous_position))
                trades.loc[trades.index[i], 'Realized PnL'] = realized_pnl - transaction_cost * abs(position_change) * execution_price

        # Update Account Size for transaction costs
        transaction_cost_amount = transaction_cost * abs(position_change) * execution_price
        trades.loc[trades.index[i], 'Account Size'] = trades['Account Size'].iloc[i-1] - transaction_cost_amount

        # Calculate and apply holding cost for the current day
        holding_cost = abs(previous_position) * execution_price * holding_cost_rate
        trades.loc[trades.index[i], 'Holding Cost'] = holding_cost
        trades.loc[trades.index[i], 'Account Size'] -= holding_cost

        # Update Unrealized PnL
        if trades['Position'].iloc[i] != 0:
            unrealized_pnl = trades['Position'].iloc[i] * (execution_price - trades['Avg Position Price'].iloc[i])
            trades.loc[trades.index[i], 'Unrealized PnL'] = unrealized_pnl
        else:
            trades.loc[trades.index[i], 'Unrealized PnL'] = 0

        # Total PnL = Cumulative Realized PnL + Unrealized PnL
        cumulative_realized_pnl = trades['Realized PnL'].cumsum().iloc[i]
        trades.loc[trades.index[i], 'Total PnL'] = cumulative_realized_pnl + trades['Unrealized PnL'].iloc[i]

        # Update Account Size for Realized and Unrealized PnL
        trades.loc[trades.index[i], 'Account Size'] = initial_capital + trades['Total PnL'].iloc[i] - holding_cost


    # Calculate Daily PnL
    trades['Daily PnL'] = trades['Total PnL'].diff().fillna(0)

    # Ensure the index is a DateTimeIndex
    trades.index = pd.to_datetime(trades.index)
    data.index = pd.to_datetime(data.index)

    weekly_pnl = pd.DataFrame()
    # Resample to weekly PnL and store in a new DataFrame
    weekly_pnl['Weekly PnL'] = trades['Daily PnL'].resample('W-MON').sum().ffill()

    # Assign the resampled Weekly PnL back to the trades DataFrame
    trades = trades.merge(weekly_pnl, left_index=True, right_index=True, how='left')

    return trades

def calculate_backtest_metrics(trades, capital, strat):
    daily_pnl = trades['Daily PnL']
    weekly_pnl = trades['Weekly PnL']

    # Calculate Z-scores to identify outliers
    z_scores = (weekly_pnl - weekly_pnl.mean()) / weekly_pnl.std()
    outliers = weekly_pnl[np.abs(z_scores) > 3]

    cumulative_pnl = trades['Total PnL']
    daily_mean_return = daily_pnl.mean()
    daily_std_dev = daily_pnl.std()
    annualized_return = (daily_mean_return * 252) / capital
    annualized_std_dev = (daily_std_dev * np.sqrt(252)) / capital
    sharpe_ratio = annualized_return / annualized_std_dev 
    annualized_return_pct = annualized_return * 100
    drawdowns = cumulative_pnl - cumulative_pnl.cummax()
    max_drawdown = drawdowns.min()
    avg_drawdown = drawdowns.mean()
    avg_position = trades['Position'].abs().mean()
    total_turnover = trades['Position'].diff().abs().sum()
    annual_turnover = (total_turnover / avg_position / len(trades)) * 252
    skewness = weekly_pnl.skew()
    kurtosis = weekly_pnl.kurtosis()
    lower_tail = daily_pnl.quantile(0.05)
    upper_tail = daily_pnl.quantile(0.95)

    metrics = pd.DataFrame({
        "Initial Capital": capital,
        "Mean Annualised Return %": annualized_return_pct,
        "Annual Standard Deviation %": annualized_std_dev * 100,
        "Annual Sharpe Ratio": sharpe_ratio,
        "Annual Turnover": annual_turnover,
        "Annual Max Drawdown %": (max_drawdown / capital) * 100,
        "Annual Avg Drawdown %": (avg_drawdown / capital) * 100,
        "Weekly Skew": skewness,
        "Weekly Kurtosis": kurtosis,
        "Lower Tail (5th percentile)": lower_tail,
        "Upper Tail (95th percentile)": upper_tail,
        "Cumulative PnL": cumulative_pnl.iloc[-1],
        "Outliers": len(outliers),
    }, index=[strat]).T

    # Format the metrics with appropriate symbols
    format_dict = {
        "Mean Annualised Return %": "{:.2f}%",
        "Annual Standard Deviation %": "{:.2f}%",
        "Annual Max Drawdown %": "{:.2f}%",
        "Annual Avg Drawdown %": "{:.2f}%",
        "Annual Sharpe Ratio": "{:.2f}",
        "Annual Turnover": "{:.2f}",
        "Weekly Skew": "{:.2f}",
        "Weekly Kurtosis": "{:.2f}",
        "Lower Tail (5th percentile)": "${:.2f}",
        "Upper Tail (95th percentile)": "${:.2f}",
        "Cumulative PnL": "${:,.2f}",
        "Outliers": "{:.0f}",
        "Initial Capital": "${:,.2f}"
    }

    for metric in metrics.index:
        metrics.loc[metric] = format_dict[metric].format(metrics.loc[metric, strat])

    return metrics, outliers

    
def calculate_position_size(capped_signals, risk_target, volatility, price, capital, instrument_weights, fx_rate, idm):
    multiplier = 1000
    weighted_signals = capped_signals.mul(instrument_weights, axis=1)
    average_signal = weighted_signals.sum(axis=1)
    position = round(average_signal * capital * idm * risk_target / (10 * multiplier * price * fx_rate * volatility), 0)
    return position

def plot_main(trades, price, pnl, composite_signal, weekly_pnl):
    fig = make_subplots(rows=5, cols=1, shared_xaxes=False, 
                        row_heights=[2, 2, 2, 2, 2], vertical_spacing=0.05)

    # Plot the underlying price
    fig.add_trace(go.Scatter(x=price.index, y=price, mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)

    # Plot the PnL
    fig.add_trace(go.Scatter(x=pnl.index, y=pnl, mode='lines', name='PNL', line=dict(color='green')), row=2, col=1)

    # Plot the Weekly PnL
    fig.add_trace(go.Scatter(x=weekly_pnl.index, y=weekly_pnl, mode='lines', name='Weekly PnL', line=dict(color='orange')), row=3, col=1)

    # Plot the Position Size
    fig.add_trace(go.Scatter(x=trades.index, y=trades['Position'], mode='lines', name='Position Size', line=dict(color='red')), row=4, col=1)

    # Plot long and short positions as filled areas
    fig.add_trace(go.Scatter(x=trades.index, y=trades['Position'] * (trades['Position'] > 0), mode='lines', fill='tozeroy', name='Long Position', fillcolor='rgba(0,255,0,0.3)', line=dict(width=0), hoverinfo='skip'), row=4, col=1)
    fig.add_trace(go.Scatter(x=trades.index, y=trades['Position'] * (trades['Position'] < 0), mode='lines', fill='tozeroy', name='Short Position', fillcolor='rgba(255,0,0,0.3)', line=dict(width=0), hoverinfo='skip'), row=4, col=1)

    # Highlight direction changes
    for i in range(1, len(trades)):
        if trades['Position'].iloc[i] != 0 and trades['Position'].iloc[i - 1] == 0:
            fig.add_shape(type="line", xref="x", yref="paper", x0=trades.index[i], x1=trades.index[i], y0=0, y1=1, line=dict(color="green" if trades['Position'].iloc[i] > 0 else "red", width=1), opacity=0.1)

    # Plot the Composite Signal
    fig.add_trace(go.Scatter(x=composite_signal.index, y=composite_signal, mode='lines', name='Composite Signal', line=dict(color='purple')), row=5, col=1)

    fig.update_layout(title='Trading Performance Overview: Price, PnL, Position Size and Signals', xaxis_title='Date', height=1500)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative PnL", row=2, col=1)
    fig.update_yaxes(title_text="Weekly PnL", row=3, col=1)
    fig.update_yaxes(title_text="Position Size", row=4, col=1)
    fig.update_yaxes(title_text="Composite Signal", row=5, col=1)

    fig.show()

def plot_hist_box(weekly_pnl):
    # Drop NaNs from weekly_pnl
    weekly_pnl = weekly_pnl.dropna()
    
    # Split into quintiles
    quintiles = pd.qcut(weekly_pnl, 5, labels=False)

    # Create DataFrame with quintiles
    df_quintiles = pd.DataFrame({'Weekly PnL': weekly_pnl, 'Quintile': quintiles})

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram of Weekly PnL", "Box Plot of Weekly PnL Quintiles"))

    # Plot histogram of Weekly PnL using Plotly
    hist_data = go.Histogram(x=weekly_pnl, nbinsx=50, name='Weekly PnL Histogram')
    fig.add_trace(hist_data, row=1, col=1)

    # Calculate density curve
    kde = gaussian_kde(weekly_pnl)
    x = np.linspace(min(weekly_pnl), max(weekly_pnl), 1000)
    density = kde(x)

    density_curve = go.Scatter(x=x, y=density * len(weekly_pnl) * (x[1] - x[0]) * 50, mode='lines', name='Density Curve')
    fig.add_trace(density_curve, row=1, col=1)

    # Plot box plot of Weekly PnL quintiles
    box_plots = []
    for quintile in range(5):
        box_plots.append(go.Box(y=df_quintiles[df_quintiles['Quintile'] == quintile]['Weekly PnL'], 
                                name=f'Quintile {quintile + 1}', boxmean=True))
    for box_plot in box_plots:
        fig.add_trace(box_plot, row=1, col=2)

    # Add vertical line at x = 0
    fig.add_shape(type="line",
                  x0=0, y0=0, x1=0, y1=1,
                  xref='x', yref='paper',
                  line=dict(color="red", width=2, dash="dash"))

    fig.update_layout(title_text="Weekly PnL Distribution", showlegend=False, height=600)

    fig.update_xaxes(title_text="Weekly PnL", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Quintiles", row=1, col=2)
    fig.update_yaxes(title_text="Weekly PnL", row=1, col=2)

    fig.show()

def backtest(ticker='BZ=F', start_date='2020-01-01', end_date=datetime.today().strftime('%Y-%m-%d'), ewmac=[4, 8, 16, 32, 64],
             risk_target=0.10, initial_capital=100000, plot=True, transaction_cost=0.1, holding_cost_rate=0.01, max_loss_sharpe=0.05):
    strat = "EWMAC_" + '_'.join([str(i) for i in ewmac])
    instrument_weights = 1
    fx_rate = 1.0
    idm = 1
    data = fetch_data(ticker, start_date, end_date)
    
    # Ensure the data index is a DateTimeIndex
    data.index = pd.to_datetime(data.index)
    
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Volatility'] = data['Log_Returns'].ewm(span=30, min_periods=0).std()
    data = calculate_ewmac_signals(data, ewmac)
    signal_cols = [i for i in data.columns if "EWM" in i]
    data["Ideal Position Size"] = calculate_position_size(data[signal_cols], risk_target, data['Volatility'], data['Close'], initial_capital, instrument_weights, fx_rate, idm)
    data['Trade Buffer'] = adjust_trade_buffer(data, max_loss_sharpe)
    data['Position Size'] = data['Ideal Position Size'].where(
        (data['Ideal Position Size'] - data['Ideal Position Size'].shift(1)).abs() > data['Trade Buffer'],
        other=np.nan
    ).ffill().fillna(0)
    
    trades = pd.DataFrame(index=data.index)
    trades['Position'] = data["Position Size"]

    trades = calculate_pnl(trades, data, initial_capital, transaction_cost, holding_cost_rate)
    backtest_metrics, outliers = calculate_backtest_metrics(trades, initial_capital, strat)
    print(backtest_metrics)

    if plot:
        plot_main(trades, data['Close'], trades['Total PnL'], data['Composite_Signal'], trades['Weekly PnL'].dropna())
        plot_hist_box(trades['Weekly PnL'])
    return trades, data

# Running the backtest
trades, data = backtest(plot=True)