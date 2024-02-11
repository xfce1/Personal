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

PATH_TO_FILES = pathlib.Path('./GC')


ContractPrices = Dict[Union[str,pd.Timestamp], pd.DataFrame]
Trades = List[Tuple[Tuple[str, str], pd.Timestamp]]


def unrealised_pnl(
        trades: Trades,
        contract_prices: ContractPrices,
    ):
    all_pnl = []
    cumulative = 0
    for (current_contract, next_contract), date in trades:
        current_prices = contract_prices[current_contract]
        next_prices = contract_prices[next_contract]

        if date not in current_prices.index or date not in next_prices.index:
            # print(f"Missing prices for {date}: Using last available prices for interpolation")
            prev_business_day = pd.offsets.BDay(-1).apply(date)
            current_open = current_prices['Open'].loc[
                prev_business_day] if prev_business_day in current_prices.index else current_prices['Open'].iloc[-1]
            next_open = next_prices['Open'].loc[
                prev_business_day] if prev_business_day in next_prices.index else next_prices['Open'].iloc[-1]
        else:
            current_open = current_prices.loc[date, 'Open']
            next_open = next_prices.loc[date, 'Open']

        pnl = next_open - current_open
        cumulative += pnl
        all_pnl.append({'Trade Date': date, 'PnL': pnl, 'Cumulative PnL': cumulative})

    pnl_df = pd.DataFrame(all_pnl)
    # print(pnl_df)
    return pnl_df
    # pass


def rolling_prices(
        contract_prices: ContractPrices,
        reference_price: str='Close',
    ):
    # Determine roll dates by calculating the lowest difference between consecutive contracts
    dates = []
    contracts = list(contract_prices.keys())

    for i, (contract, price) in enumerate(contract_prices.items()):
        current_prices = price
        try:
            next_prices = contract_prices.get(list(contract_prices.keys())[i+1])
            common_dates = set(list(current_prices.index)).intersection(set(list(next_prices.index)))

            current_prices = current_prices[current_prices.index.isin(common_dates)]
            next_prices = next_prices[next_prices.index.isin(common_dates)]

            absolute_diff = abs(next_prices["Close"] - current_prices["Close"])
            rolling_date = absolute_diff.idxmin()
            dates.append(rolling_date)
        except IndexError:
            print("Last contract reached")
            continue
    # print(len(dates), len(contracts))
    # print(dates)


def calculate_basis(
        contract_prices: ContractPrices,
        reference_price: str='Close',
    ):
    pass


def simulate_contract_prices(
        start_date: pd.Timestamp,
        last_trade_dates: List[pd.Timestamp],
        starting_spot_price: float=2000.0,
        mean_trend: float=0.0,
        std: float=50.0,
        mean_basis: float=76.0,
    ) -> ContractPrices:
    pass


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

    def test_unrealised_pnl(self):
        """
        A futures contract is a contract between a buyer (seller) to purchase (deliver) a specified
        number of the underlying at a future point in time.

        Exposure to gold commodity prices may be implemented using Futures. In order for the position
        to have constant level of exposure (in quantity of gold terms) through time, it is necessary
        to 'roll' your futures position in to the next contract as it approaches expiry. For example
        the following dictionary gives the last trade date for Gold futures contracts in 2023:

            {'GCH23': Timestamp('2023-03-27 00:00:00'),
             'GCJ23': Timestamp('2023-04-26 00:00:00'),
             'GCK23': Timestamp('2023-05-26 00:00:00'),
             'GCM23': Timestamp('2023-06-27 00:00:00'),
             'GCN23': Timestamp('2023-07-26 00:00:00'),
             'GCQ23': Timestamp('2023-08-29 00:00:00'),
             'GCU23': Timestamp('2023-09-26 00:00:00'),
             'GCV23': Timestamp('2023-10-26 00:00:00'),
             'GCX23': Timestamp('2023-11-28 00:00:00'),
             'GCZ23': Timestamp('2023-12-27 00:00:00')}

        Given a list of trade date to contract pairs, compute the unrealised profit and loss of
        1 unit of the gold contract for a full year whilst 'rolling' it.

        For example, on the 2023-03-24, we have a position in GCH23 and need to roll into GCJ23. The
        following table shows the OHLC prices for the two contracts.

            GCH23  Open        1991.7
                   High        1995.4
                   Low         1985.5
                   Close       1985.5
                   Volume        14.0
            GCJ23  Open        1996.1
                   High        2006.5
                   Low         1977.7
                   Close       1981.0
                   Volume    212721.0

        """

        list_of_trade_dates = [
             (('GCH23', 'GCJ23'), pd.Timestamp('2023-03-20 00:00:00')),
             (('GCJ23', 'GCK23'), pd.Timestamp('2023-04-22 00:00:00')),
             (('GCK23', 'GCM23'), pd.Timestamp('2023-05-19 00:00:00')),
             (('GCM23', 'GCN23'), pd.Timestamp('2023-06-26 00:00:00')),
             (('GCN23', 'GCQ23'), pd.Timestamp('2023-07-17 00:00:00')),
             (('GCQ23', 'GCU23'), pd.Timestamp('2023-08-25 00:00:00')),
             (('GCU23', 'GCV23'), pd.Timestamp('2023-09-18 00:00:00')),
             (('GCV23', 'GCX23'), pd.Timestamp('2023-10-18 00:00:00')),
             (('GCX23', 'GCZ23'), pd.Timestamp('2023-11-19 00:00:00'))
        ]

        unrealised_pnl_ = unrealised_pnl(
                list_of_trade_dates,
                self.contract_prices,
            )

        self.assertIsInstance(unrealised_pnl_, pd.DataFrame)

    def test_rolling_prices(self):
        """
        calculate the price series from the given futures contracts. prices should be:
          - stiched: This describes that the resulant dataframe should only contain
                one time series of prices. Rolls should only happen when there is data for
                a contract. The most basic example, you would not be able to roll the contract after it
                has expired, or if there was a missing price on that day.
          - backwards-adjusted: Everytime there is a roll there is a jump in price due to the basis.
                You should adjust your price series such that there is no jump in prices and
                the last price matches the true price of the latest contract.
        how does your roll algorithm compare in PnL terms to what was calculated previously?
        why do you think it is, and what economical considerations would you make?
        """
        adjusted_prices = rolling_prices(self.contract_prices)

        self.assertIsInstance(adjusted_prices, pd.DataFrame)

    def test_calculate_basis(self):
        """
        Future prices are closely correlated with the spot price of the underlying (e.g. CO Comdty
        correlates with the spot price of Brent). The difference between future price and spot is know
        as basis, typically wider away from expiry and then 0 at expiry (no arbitrage opportunity).
        The basis can be positive (contango) or negative (backwardation), describing how a future contract holder
        values it. For example, a holder of an S&P future will discount the expected dividend,
        while a live cattle future holder will have to pay for the maintanance of cows (feed, shelter, etc.)
        produce the price series for one spot and a minimum of future 3 contracts for both cases
        summary: contango 1 spot, 3 futures - backwardation 1 spot, 3 futures (8 total).
        you can assume that both prices and basis have their own volatility (std dev) and mean.

        For this exercise, calculate the basis for each contract, using the earliest expiring contract
        as the base level. For example on the '2023-03-17', the following contract prices and basis are below.

                    Close	Basis
            GCH23	1985.1	0.0
            GCJ23	1993.7	8.6
            GCK23	2000.9	15.8
            GCM23	2009.8	24.7
            GCN23	NaN     NaN
            GCQ23	2027.1	42.0
            GCU23	NaN     NaN
            GCV23	2039.3	54.2
            GCX23	NaN     NaN
            GCZ23	2057.5	72.4

        This example, exhibits a positive basis and therefore in contango.
        """

        basis = calculate_basis(self.contract_prices, reference_price='Close')

        self.assertIsInstance(basis, pd.DataFrame)

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
            * std_basis_horizon
                The volatility of the basis of a single day
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
