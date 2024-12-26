import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Long only 1/n portfolio
        """
    )
    return


@app.cell
def __():
    import pandas as pd

    pd.options.plotting.backend = "plotly"

    import yfinance as yf

    from cvx.simulator.builder import builder
    from cvx.simulator.grid import resample_index

    return builder, pd, resample_index, yf


@app.cell
def __(yf):
    data = yf.download(
        tickers="SPY AAPL GOOG MSFT",  # list of tickers
        period="10y",  # time period
        interval="1d",  # trading interval
        prepost=False,  # download pre/post market hours data?
        repair=True,
    )  # repair obvious price errors e.g. 100x?
    return (data,)


@app.cell
def __(data):
    prices = data["Adj Close"]
    return (prices,)


@app.cell
def __():
    capital = 1e6
    return (capital,)


@app.cell
def __(builder, capital, prices, state, time):
    b = builder(prices=prices, initial_cash=capital)
    for _time, _state in b:
        b[time[-1]] = 0.25 * state.nav / state.prices
    return (b,)


@app.cell
def __(b):
    portfolio = b.build()
    portfolio.profit.cumsum().plot()
    return (portfolio,)


@app.cell
def __(portfolio):
    portfolio.nav.plot()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Rebalancing
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Usually we would not execute on a daily basis but rather rebalance every week, month or quarter.
        There are two approaches to deal with this problem in cvxsimulator.

        * Resample the existing daily portfolio (helpful to see effect of your hesitated trading)
        * Trade only on days that are within a predefined grid (most flexible if you have a rather irregular grid)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Resample an existing portfolio
        """
    )
    return


@app.cell
def __(portfolio):
    portfolio_resampled = portfolio.resample(rule="M")
    return (portfolio_resampled,)


@app.cell
def __(pd, portfolio, portfolio_resampled):
    frame = pd.DataFrame(
        {"original": portfolio.nav, "monthly": portfolio_resampled.nav}
    )
    frame
    return (frame,)


@app.cell
def __(portfolio_resampled):
    print(portfolio_resampled.stocks)
    return


@app.cell
def __(frame):
    # almost hard to see that difference between the original and resampled portfolio
    frame.plot()
    return


@app.cell
def __(portfolio_resampled):
    # number of shares traded
    portfolio_resampled.trades_stocks.iloc[1:].plot()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Trade only days in predefined grid
        """
    )
    return


@app.cell
def __(builder, capital, prices, resample_index, state, time):
    b_1 = builder(prices=prices, initial_cash=capital)
    grid = resample_index(prices.index, rule="M")
    for _time, _state in b_1:
        if _time[-1] in grid:
            b_1[time[-1]] = 0.25 * state.nav / state.prices
        else:
            b_1[time[-1]] = b_1[time[-2]]
    portfolio_1 = b_1.build()
    return b_1, grid, portfolio_1


@app.cell
def __(portfolio_1):
    portfolio_1.nav.plot()
    return


@app.cell
def __(portfolio_1):
    portfolio_1.turnover.iloc[1:].plot()
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Why not resampling the prices?
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        I don't believe in bringing the prices to a monthly grid. This would render it hard to construct signals
        given the sparse grid. We stay on a daily grid and trade once a month.

        """
    )
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
