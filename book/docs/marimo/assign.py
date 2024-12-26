import marimo

__generated_with = "0.9.27"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # Reset a large matrix
        """
    )
    return


@app.cell
def __():
    import numpy as np

    return (np,)


@app.cell
def __(np):
    a = np.random.randn(5000, 5000)
    _b = np.random.randn(5000, 5000)
    return (a,)


@app.cell
def __(a):
    _b = 0 * a
    return


@app.cell
def __(a, np):
    _b = np.zeros_like(a)
    return


@app.cell
def __(np):
    _b = np.zeros((5000, 5000))
    return


@app.cell
def __():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
