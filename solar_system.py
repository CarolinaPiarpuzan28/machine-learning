'''
Script description: Get all data about solar system.
'''
import os
import requests

def clear():
    os.system('clear')

API_BASE = "https://api.le-systeme-solaire.net/rest/bodies/"


FILTERS = {
    "1": ("bodyType", "Planet"),    # Planetas
    "2": ("bodyType", "Moon"),      # Lunas
    "3": ("bodyType", "Star"),      # Estrellas
    "4": ("bodyType", "Asteroid"),  # Asteroides
    "5": ("bodyType", "Comet"),     # Cometas
}

def fetch_bodies(field: str, value: str):
    """Consulta a la API con filtros"""
    params = {"filter[]": f"{field},eq,{value}"}
    try:
        r = requests.get(API_BASE, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("bodies", [])
    except requests.exceptions.RequestException as e:
        print(f"API error: {e}")
        return []

def show_list(title: str, bodies: list, limit: int = 10):
    """Muestra la lista de cuerpos celestes con algunos datos extra"""
    print(f"\n--- {title} ---")
    if not bodies:
        print("No data found.\n")
        return
    for body in bodies[:limit]:
        name = body.get("englishName") or body.get("name") or "Unknown"
        btype = body.get("bodyType", "")
        gravity = body.get("gravity", "N/A")
        print(f"- {name} [{btype}] | Gravity: {gravity}")
    print()

def menu():
    while True:
        print("::: COMETS INFORMATION :::")
        print("::: SOLAR SYSTEM MENU :::")
        print("[1]. Planets")
        print("[2]. Moons")
        print("[3]. Stars")
        print("[4]. Asteroid")
        print("[5]. Comets")
        print("[6]. Exit")
        choice = input("Choose an option: ").strip()
        clear()

        if choice in FILTERS:
            field, value = FILTERS[choice]
            items = fetch_bodies(field, value)
            titles = {
                "1": "PLANETS",
                "2": "MOONS",
                "3": "STARS",
                "4": "ASTEROIDS",
                "5": "COMETS",
            }
            show_list(titles[choice], items)
        elif choice == "6":
            print("Exiting program...")
            break
        else:
            print("Invalid option, try again!\n")


if __name__ == "__main__":
    clear()
    menu()
