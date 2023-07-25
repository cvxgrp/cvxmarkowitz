# -*- coding: utf-8 -*-
import fire
import requests


def cli(metric: str, latitude: float = 37.4419, longitude: float = -122.143) -> None:
    """
    Get the current weather for a given metric

    Parameters
    ----------
    metric : str
        The metric to get the current weather for
    latitude : float, optional
        The latitude to get the current weather for, by default 37.4419
    longitude : float, optional
        The longitude to get the current weather for, by default -122.143
    """
    url = "https://api.open-meteo.com/v1/forecast"
    url = f"{url}?latitude={str(latitude)}&longitude={str(longitude)}&current_weather=true"
    r = requests.get(url)

    if r.status_code == 200:
        if metric in r.json()["current_weather"]:
            x = r.json()["current_weather"][metric]
            return x
        else:
            raise ValueError("Metric not supported!")
    else:
        raise ConnectionError("Open-Meteo is down!")


def main():  # pragma: no cover
    """
    Run the CLI using Fire
    """
    fire.Fire(cli)
