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
            print(r.json()["current_weather"][metric])
        else:
            print("Metric not supported!")
    else:
        print("Open-Meteo is down!")


if __name__ == "__main__":
    fire.Fire(cli)
