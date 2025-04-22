import requests
import os

BASE_URL = "https://api.fda.gov/drug/label.json"

def search_fda_label(ae_name: str) -> dict:
    """
    Search for FDA labels using the FDA API.
    
    Args:
        ae_name (str): The name of the adverse effect to search for.
    Returns:
        dict: The response from the FDA API.
    """
    params = {
        'search': f'adverse_reactions_table:"{ae_name}"',
        'limit': 1000
    }
    response = requests.get(BASE_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}")
        return {}
    
# test usage
res = search_fda_label("nausea")
print(res)