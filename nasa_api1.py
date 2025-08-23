'''
Script description: Get and read data from NASA API about space
NASA API: https://api.nasa.gov/
Dev: carolina.
https://api.nasa.gov/neo/rest/v1/neo/3726712?api_key={API_KEY_HERE}
'''

import os
import requests

os.system('clear')


def get_nasa_data(api_key):
    print("::: NASA INFORMATION :::")
    url = f"https://api.nasa.gov/neo/rest/v1/neo/3726712?api_key={api_key}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extraer información 
        name = data.get("name", "No disponible")
        magnitude = data.get("absolute_magnitude_h", "No disponible")
        diameter_km = data["estimated_diameter"]["kilometers"]["estimated_diameter_max"]
        diameter_ft = data["estimated_diameter"]["feet"]["estimated_diameter_max"]

        # Mostrar en consola
        print(f"Nombre del cometa: {name}")
        print(f"Magnitud absoluta h: {magnitude}")
        print(f"Diámetro máximo estimado (KM): {diameter_km}")
        print(f"Diámetro máximo estimado (FT): {diameter_ft}")

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
    except KeyError as ke:
        print(f"Error de clave en JSON: {ke}")


# API Key de NASA
NASA_API_KEY = "cqPwMqZhBpj7JZ4zmbHtuMApSi7aEdiyt4QMJEzS"

get_nasa_data(NASA_API_KEY)
