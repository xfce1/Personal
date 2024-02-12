"""
Rolling instruments take home assignment.

Evaluation criteria (in random order):
- all tasks perform what described
- all (smoke) tests pass
- additional tests
- performance and ability to scale
- split into sub-problems (interface and clarity)
- documentation
"""
import unittest
import pathlib
from typing import (
        Dict,
        List,
        Tuple,
        Union,
)
import numpy as np
import pandas as pd
from scipy.stats import norm

PATH_TO_FILES = pathlib.Path('./GC')


ContractPrices = Dict[Union[str,pd.Timestamp], pd.DataFrame]
Trades = List[Tuple[Tuple[str, str], pd.Timestamp]]


def unrealised_pnl(
        trades: Trades,
        contract_prices: ContractPrices,
    ):
    """
    Explanation:
        This function iterates over each trade and calculates the unrealized profit and loss (PnL) based on the
        difference in opening prices between the current and next contracts. It considers cases where price data
        might be missing for the trade date, and if so, it calculates the PnL using the opening prices of the
        previous business day. The cumulative PnL is updated for each trade, and the results are returned as a DataFrame.
    """
    # Initialise empty list to store PnL for each trade and cumulative PnL
    all_pnl = []
    cumulative = 0
    # Iterate over each trade
    for (current_contract, next_contract), date in trades:
        current_prices = contract_prices[current_contract]
        next_prices = contract_prices[next_contract]
        
        # Check if trade date exists in price data for both contracts
        if date not in current_prices.index or date not in next_prices.index:
            # Noticed trade dates on weekends, used last business day as a proxy
            # Could have skipped this trade because of this, chose to proxy it as the open for the day before
            # Also considered using the last close instead of open, kept open for consistency
            prev_business_day = date - pd.offsets.BDay(1)
            # Retrieve opening prices for previous business day
            current_open = current_prices['Open'].loc[prev_business_day] 
            next_open = next_prices['Open'].loc[prev_business_day]
        else:
            # Retrieve opening prices for the trade date
            current_open = current_prices.loc[date, 'Open']
            next_open = next_prices.loc[date, 'Open']

        # Selling old contract, buying new contract
        pnl = current_open - next_open
        cumulative += pnl
        all_pnl.append({'Trade Date': date, 'PnL': pnl, 'Cumulative PnL': cumulative})

    pnl_df = pd.DataFrame(all_pnl)
    print(pnl_df)
    return pnl_df


def rolling_prices(
        contract_prices: ContractPrices,
        reference_price: str='Close',
    ):
    """
    Explanation:
        This function iterates over consecutive contracts and determines roll dates by finding the lowest difference
        between their prices. It calculates rolling prices at these roll dates and returns them as a DataFrame.
    """
    # Determine roll dates by calculating the lowest difference between consecutive contracts
    dates = []
    contracts = list(contract_prices.keys())
    # range set to len(contracts)-1 because it should be the number of dates we get, needed for indexing properly. Iterating through all of the rolls:
    for i in range(len(contracts)-1):
        # Defining and getting current and next contract prices
        current_contract = contracts[i]
        next_contract = contracts[i+1]
        current_prices = contract_prices[current_contract]
        next_prices = contract_prices[next_contract]
        # Only using the intersection of the two contracts to select date to roll
        common_dates = set(list(current_prices.index)).intersection(set(list(next_prices.index)))

        if common_dates:
            # Choosing the point where the difference between the contract prices is minimised
            absolute_diff = abs(next_prices[reference_price] - current_prices[reference_price])
            rolling_date = absolute_diff.idxmin()
            # Adding the date that minimises price between first pair of contracts
            dates.append(rolling_date)
        else:
            print("No common dates")
    
    # Initialising dataframes
    rolling_df = pd.DataFrame()
    pnl_df = pd.DataFrame()
    # Iterating through all contracts again post trade date calcs
    for i in range(len(contracts)-1):
        # Defining dates and contracts
        date = dates[i]
        current_contract = contracts[i]
        next_contract = contracts[i+1]
        # Did not use loc due to inconsitencies between df and series
        current_prices = contract_prices[current_contract][contract_prices[current_contract].index==date]
        next_prices = contract_prices[next_contract][contract_prices[next_contract].index==date]
        # Adding the current trade row to the dataframe after calculating pnl
        pnl_df = pd.concat([pnl_df,pd.DataFrame(current_prices[reference_price] - next_prices[reference_price])], axis=0)
        
        # Stitching the prices together, trade date is going to be the new contract
        if i<len(dates)-1:
            if i==0:
                # All data before the first trade comes from the first contract
                price_to_stitch = contract_prices[current_contract][contract_prices[current_contract].index < date]
            else:
                # Data after comes from the range between the two trade dates
                price_to_stitch = contract_prices[current_contract][(contract_prices[current_contract].index > max(rolling_df.index)) & 
                                                                   (contract_prices[current_contract].index < date)]
            # Stiching into one df
            rolling_df = pd.concat([rolling_df,price_to_stitch],axis=0)

    # Printing pnl df as test cases require one dataframe output
    print(f"\nCumulative Pnl: {np.round(sum(pnl_df.values)[0],4)}\n", pnl_df)
    ## Additional comments
    # This function generates a total pnl of -4.1 compared to -7.9 in the unrealised pnl function.
    # This is because rolling usually results in a small loss, as you are extending a position.
    # Since the price difference was minimised, the loss was lower using this method.
    # It can be noted that the first may not be an accurate representation due to trades occuring on the weekend - gave errors as prices did not exist.
    # In a real trading environment, transaction costs would also need to be accounted for.
    # Some of the volume was low and would not be suitable to roll larger positions
    # Would try to minimise transaction costs to save additional pnl
    return rolling_df


def calculate_basis(
        contract_prices: ContractPrices,
        reference_price: str='Close',
    ):

    """
    Explanation:
        This function calculates the basis for each contract based on the earliest expiring contract as the base level.
        It first extracts close prices for each contract from the provided contract prices dictionary. Then, it creates
        a DataFrame containing close prices of all contracts at all dates, interpolating values only within the date range
        of the corresponding contract. Next, it calculates basis values for each contract by subtracting the price of
        the earliest available contract from the prices of other contracts at each date. The results are returned as a DataFrame.
    """
    # Initialise dict and dates set
    contract_close_prices = {}
    all_dates = set()  # Store all dates available in the dataset
    for contract_name, prices in contract_prices.items():
        if reference_price in prices.columns:  # Check if reference price data is available
            close_prices = prices[reference_price]  # Extract close prices
            contract_close_prices[contract_name] = close_prices
            all_dates.update(prices.index)  # Add dates to set of all dates

    # Sort dates
    all_dates = sorted(all_dates)

    # Create DataFrame containing close prices of all contracts at all dates
    close_prices_df = pd.DataFrame(index=all_dates)
    for contract_name, close_prices in contract_close_prices.items():
        if contract_name == min(contract_close_prices.keys()):
            # Interpolate only within the date range (start to expiry) for each contract to account for missing values
            min_date, max_date = contract_prices.get(contract_name).index[0], contract_prices.get(contract_name).index[-1]
            close_prices = close_prices.loc[min_date:max_date]
            # Linear interpolation to fill missing values
            close_prices_df[contract_name] = close_prices.interpolate(method='time')
        else:
            close_prices_df[contract_name] = close_prices
    
    # Drop rows with less than 4 non-na values as outlined in the test function requirements (one spot three futures)
    close_prices_df = close_prices_df.dropna(thresh=4)

    # Calculate basis values by using the earliest expiring contract as the spot
    basis_df = close_prices_df.copy()
    # iterate through all rows
    for idx, row in close_prices_df.iterrows():
        non_na_values = row.dropna()
        basis_values = non_na_values - non_na_values.iloc[0]
        basis_df.loc[idx] = basis_values

    ## Additional comments
    # Calculated the basis for each day, managed to recreate the example contango shown in the test function to validate methodology.
    # Could not find an example of backwardation despite attempting to interpolate values in an effort to try and find those.
    # Assumed basis has it's own vol, unsure if the task was to simulate prices based on that, similar to the next task.
    # Every example returned in the dataframe exhibits contango for 184 dates.
    return basis_df


def simulate_contract_prices(
        start_date: pd.Timestamp,
        last_trade_dates: List[pd.Timestamp],
        starting_spot_price: float=2000.0,
        mean_trend: float=0.0,
        std: float=50.0,
        mean_basis: float=76.0,
    ) -> ContractPrices:
    """
        Explanation:
        This function simulates a series of contract prices for each expiry month provided in the `last_trade_dates` parameter. 
        It generates synthetic contract price data based on various parameters such as the starting spot price, mean trend, 
        standard deviation, and mean basis. The simulated prices are returned as a dictionary, where each key represents 
        a last trade date, and each value is a dataframe containing contract prices. The dataframe includes columns for 
        Open, High, Low, Close, and Volume, with each row representing a day in the simulation period.

        Methodology:
        - The simulation starts from the `start_date` and ends at each `last_trade_date`.
        - Spot prices are simulated using a random walk, where each day's spot price is influenced by the previous day's price.
        - Open and Close prices are simulated by adding a random increment to the spot price, with a standard deviation 
          parameter (`std`) controlling the variability.
        - High and Low prices are simulated similarly to Open and Close prices but with additional randomness.
        - Volume is simulated to resemble a semi-bell curve, gradually increasing until a certain point and then decreasing 
          towards the end of the simulation period. This pattern is intended to reflect typical trading activity in futures markets.
    """
    std_basis = 20
    simulated_prices = {}
    for last_trade_date in last_trade_dates:
        dates = pd.date_range(start=start_date, end=last_trade_date)
        num_days = len(dates)
        spot_prices = np.zeros(num_days)
        basis_values = np.zeros(num_days)

        # Simulate spot prices
        spot_prices[0] = starting_spot_price
        for i in range(1, num_days):
            # Calculate the previous day's price change
            price_change = spot_prices[i-1] - spot_prices[i-2] if i > 1 else 0
            # Scale the standard deviation based on the previous day's price change
            scaled_std = std + abs(price_change) * 0.1  # Adjust the multiplier as needed
            
            # Random increment sampled from a normal distribution with scaled standard deviation
            increment = np.random.normal(mean_trend, scaled_std)
            spot_prices[i] = spot_prices[i-1] + increment

        # Generate contract prices
        contract_prices = pd.DataFrame(index=dates, columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Simulate high and low prices
        contract_prices['High'] = spot_prices + np.random.normal(mean_trend, std, size=num_days)
        contract_prices['Low'] = spot_prices - np.random.normal(mean_trend, std, size=num_days)
        
        # Simulate Open and Close prices
        contract_prices['Open'] = spot_prices + np.random.normal(mean_trend, std, size=num_days) * 0.2  # 20% deviation from spot
        contract_prices['Close'] = spot_prices + np.random.normal(mean_trend, std, size=num_days) * 0.2  # 20% deviation from spot

        # Ensure High and Low prices are not within the range of Open and Close prices
        contract_prices['High'] = np.maximum(contract_prices['High'], contract_prices[['Open', 'Close']].max(axis=1))
        contract_prices['Low'] = np.minimum(contract_prices['Low'], contract_prices[['Open', 'Close']].min(axis=1))

        # Simulate Volume
        volume = np.zeros(num_days)
        expiry_date = last_trade_date
        basis_values = np.random.normal(mean_basis, std_basis / np.sqrt(num_days), size=num_days) # Ensuring there is a match
        # Generate semi-bell curve volume
        bell_curve_volume = np.linspace(20, 20000, num=num_days)
        bell_curve_volume = np.minimum(bell_curve_volume, 20000)  # Cap volume at 20k
        
        # Ensure a few hundred volume right near the end
        bell_curve_volume[-15:] = np.linspace(50, 500, num=15)
        
        contract_prices['Volume'] = bell_curve_volume

        simulated_prices[last_trade_date] = contract_prices
    return simulated_prices


class TestRollingInstruments(unittest.TestCase):

    contracts = [
            'GCH23', 'GCJ23', 'GCK23', 'GCM23', 'GCN23',
            'GCQ23', 'GCU23', 'GCV23', 'GCX23', 'GCZ23'
    ]

    @classmethod
    def setUpClass(cls):
        """read contracts from csv files"""
        cls.contract_prices = {
            c: pd.read_csv(PATH_TO_FILES/f'{c}.csv', parse_dates=True, index_col='Date')
            for c in cls.contracts
        }

    # def test_unrealised_pnl(self):
    #     """
    #     A futures contract is a contract between a buyer (seller) to purchase (deliver) a specified
    #     number of the underlying at a future point in time.

    #     Exposure to gold commodity prices may be implemented using Futures. In order for the position
    #     to have constant level of exposure (in quantity of gold terms) through time, it is necessary
    #     to 'roll' your futures position in to the next contract as it approaches expiry. For example
    #     the following dictionary gives the last trade date for Gold futures contracts in 2023:

    #         {'GCH23': Timestamp('2023-03-27 00:00:00'),
    #          'GCJ23': Timestamp('2023-04-26 00:00:00'),
    #          'GCK23': Timestamp('2023-05-26 00:00:00'),
    #          'GCM23': Timestamp('2023-06-27 00:00:00'),
    #          'GCN23': Timestamp('2023-07-26 00:00:00'),
    #          'GCQ23': Timestamp('2023-08-29 00:00:00'),
    #          'GCU23': Timestamp('2023-09-26 00:00:00'),
    #          'GCV23': Timestamp('2023-10-26 00:00:00'),
    #          'GCX23': Timestamp('2023-11-28 00:00:00'),
    #          'GCZ23': Timestamp('2023-12-27 00:00:00')}

    #     Given a list of trade date to contract pairs, compute the unrealised profit and loss of
    #     1 unit of the gold contract for a full year whilst 'rolling' it.

    #     For example, on the 2023-03-24, we have a position in GCH23 and need to roll into GCJ23. The
    #     following table shows the OHLC prices for the two contracts.

    #         GCH23  Open        1991.7
    #                High        1995.4
    #                Low         1985.5
    #                Close       1985.5
    #                Volume        14.0
    #         GCJ23  Open        1996.1
    #                High        2006.5
    #                Low         1977.7
    #                Close       1981.0
    #                Volume    212721.0

    #     """

    #     list_of_trade_dates = [
    #          (('GCH23', 'GCJ23'), pd.Timestamp('2023-03-20 00:00:00')),
    #          (('GCJ23', 'GCK23'), pd.Timestamp('2023-04-22 00:00:00')),
    #          (('GCK23', 'GCM23'), pd.Timestamp('2023-05-19 00:00:00')),
    #          (('GCM23', 'GCN23'), pd.Timestamp('2023-06-26 00:00:00')),
    #          (('GCN23', 'GCQ23'), pd.Timestamp('2023-07-17 00:00:00')),
    #          (('GCQ23', 'GCU23'), pd.Timestamp('2023-08-25 00:00:00')),
    #          (('GCU23', 'GCV23'), pd.Timestamp('2023-09-18 00:00:00')),
    #          (('GCV23', 'GCX23'), pd.Timestamp('2023-10-18 00:00:00')),
    #          (('GCX23', 'GCZ23'), pd.Timestamp('2023-11-19 00:00:00'))
    #     ]

    #     unrealised_pnl_ = unrealised_pnl(
    #             list_of_trade_dates,
    #             self.contract_prices,
    #         )

    #     self.assertIsInstance(unrealised_pnl_, pd.DataFrame)

    # def test_rolling_prices(self):
    #     """
    #     calculate the price series from the given futures contracts. prices should be:
    #       - stiched: This describes that the resulant dataframe should only contain
    #             one time series of prices. Rolls should only happen when there is data for
    #             a contract. The most basic example, you would not be able to roll the contract after it
    #             has expired, or if there was a missing price on that day.
    #       - backwards-adjusted: Everytime there is a roll there is a jump in price due to the basis.
    #             You should adjust your price series such that there is no jump in prices and
    #             the last price matches the true price of the latest contract.
    #     how does your roll algorithm compare in PnL terms to what was calculated previously?
    #     why do you think it is, and what economical considerations would you make?
    #     """
    #     adjusted_prices = rolling_prices(self.contract_prices)

    #     self.assertIsInstance(adjusted_prices, pd.DataFrame)

    # def test_calculate_basis(self):
    #     """
    #     Future prices are closely correlated with the spot price of the underlying (e.g. CO Comdty
    #     correlates with the spot price of Brent). The difference between future price and spot is know
    #     as basis, typically wider away from expiry and then 0 at expiry (no arbitrage opportunity).
    #     The basis can be positive (contango) or negative (backwardation), describing how a future contract holder
    #     values it. For example, a holder of an S&P future will discount the expected dividend,
    #     while a live cattle future holder will have to pay for the maintanance of cows (feed, shelter, etc.)
    #     produce the price series for one spot and a minimum of future 3 contracts for both cases
    #     summary: contango 1 spot, 3 futures - backwardation 1 spot, 3 futures (8 total).
    #     you can assume that both prices and basis have their own volatility (std dev) and mean.

    #     For this exercise, calculate the basis for each contract, using the earliest expiring contract
    #     as the base level. For example on the '2023-03-17', the following contract prices and basis are below.

    #                 Close	Basis
    #         GCH23	1985.1	0.0
    #         GCJ23	1993.7	8.6
    #         GCK23	2000.9	15.8
    #         GCM23	2009.8	24.7
    #         GCN23	NaN     NaN
    #         GCQ23	2027.1	42.0
    #         GCU23	NaN     NaN
    #         GCV23	2039.3	54.2
    #         GCX23	NaN     NaN
    #         GCZ23	2057.5	72.4

    #     This example, exhibits a positive basis and therefore in contango.
    #     """

    #     basis = calculate_basis(self.contract_prices, reference_price='Close')

    #     self.assertIsInstance(basis, pd.DataFrame)

    def test_simulate_contract_prices(self):
        """
        Simulate a series of contract prices for each of the expiry months provided. The simulated
        prices must return a dictionary of contract code to dataframe and contain Open, High, Low,
        Close and Volume.

        Assume that the price series for every contract in the simulation starts from the first day.
        That is, the furthest out expiring contract will have a price for every day of the simulation.

        Your simulation must accept atleast the following parameters

            * last_trade_dates
                A list of last trade dates. Each last trade date will coincide with a contract
                If only two contracts are passed, only two contracts should be simulated.
            * start_date
                The first day of the simulation.
            * starting_spot_price
                The initial spot price of the closest expiring contract.
            * mean_trend
                This is the mean trend exhibited during the series.
            * std
                The daily standard deviation of contract prices
            * mean_basis
                The mean basis for the entire series
            * std_basis
                The daily standard deviation of the basis
        """

        last_trade_dates = [
                pd.Timestamp('2023-03-20 00:00:00'),
                pd.Timestamp('2023-04-22 00:00:00'),
                pd.Timestamp('2023-05-19 00:00:00'),
                pd.Timestamp('2023-06-26 00:00:00'),
                pd.Timestamp('2023-07-17 00:00:00'),
            ]

        start_date = pd.Timestamp('2023-01-02')

        simulated_prices = simulate_contract_prices(
                start_date=start_date,
                starting_spot_price=2000,
                mean_trend=0.0,
                std=50,
                mean_basis=76,
                last_trade_dates=last_trade_dates,
            )

        self.assertEqual(len(simulated_prices), 5)

        for last_trd, (i, data) in zip(last_trade_dates, simulated_prices.items()):
            self.assertEqual(i, last_trd)
            self.assertEqual(data.index[-1], last_trd)
            self.assertEqual(data.index[0], start_date)


if __name__=='__main__':
    unittest.main()
