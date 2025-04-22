"""
Last edited: 2025-04-21
Author: Albert Cao
Description: Extracting the adverse effects from drug labels of FDA approved drugs.
The drug labels are downloaded from https://open.fda.gov/apis/drug/label/download/.
"""

import bs4
import json
import re
from openai import OpenAI
import pandas as pd
import numpy as np

client = OpenAI()

instructions = """
You are a helpful assistant that extracts the adverse effects from drug labels of FDA approved drugs.
The drug labels are downloaded from https://open.fda.gov/apis/drug/label/download/.
The drug labels are in JSON format and contain a field called 'adverse_reactions_table'.
The 'adverse_reactions_table' field contains an HTML table with the adverse effects.

Your task is to extract the adverse effects from the HTML table and return them in a structured format.
The structured format should be a dictionary with (key, value) pairs, where the key
is the name of the adverse effect and the value is the percentage of patients that experienced that adverse effect.
The percentage should be a float between 0 and 100.

The adverse effects may have different names in different drug labels, so you should make sure to
normalize the names of the adverse effects.
The percentage may be in different formats, such as '29 %', '<1 %', or '0.5 %'.
You should also make sure to handle these different formats and convert them to a float between 0 and 100.

You will receive an HTML table as a string.

Make sure to ONLY return the structured data and nothing else.
The structured data should be in a JSON format.
"""

def parse_adverse_reactions_table_llm(html_string: str) -> dict:
    """
    Parse an HTML table of adverse reactions and return structured data using OpenAI API.

    Args:
        html_string (str): String containing HTML table.
    Returns:
        dict: { "adverse_effect_1": percentage_1, "adverse_effect_2": percentage_2, ... }
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": instructions},
            {"role": "user", "content": html_string}
        ],
        max_tokens=500
    )
    json_response = response.choices[0].message.content
    # strip the ```json and ``` from the start and end of the response
    json_response = json_response.strip().lstrip('```json').rstrip('```')
    dict_response = json.loads(json_response)
    return dict_response

def parse_adverse_reactions_table(html_string: str) -> dict:
    """
    Parse an HTML table of adverse reactions and return structured data.

    Args:
        html_string (str): String containing HTML table.

    Returns:
        dict: {
            'columns': [col1, col2, ...],
            'data': [
                {col1: val1, col2: val2, ...},
                ...
            ]
        }
    """
    # Parse HTML
    soup = bs4.BeautifulSoup(html_string, 'html.parser')
    table = soup.find('table')
    if not table:
        raise ValueError("No <table> found in provided HTML.")

    # header row: use <thead> if present, else first <tr> in <tbody>
    if table.thead and table.thead.find('tr'):
        header_row = table.thead.find('tr')
    else:
        header_row = table.find('tbody').find('tr')
    headers = [cell.get_text(strip=True) for cell in header_row.find_all(['th', 'td'])]

    # go over body rows (skip header_row)
    data_rows = []
    for tr in table.find('tbody').find_all('tr'):
        if tr is header_row:
            continue
        cells = tr.find_all('td')
        # skip rows that don't match header length
        if len(cells) != len(headers):
            continue
        row_dict = {}
        for header, cell in zip(headers, cells):
            text = cell.get_text(strip=True)
            # try to parse percentages like '29 %' or '<1 %'
            pct_match = re.match(r'^<?(\d+(?:\.\d+)?)\s*%$', text)
            if pct_match:
                row_dict[header] = float(pct_match.group(1))
            else:
                # try numeric without %
                num_match = re.match(r'^<?(\d+(?:\.\d+)?)$', text)
                if num_match:
                    row_dict[header] = float(num_match.group(1))
                else:
                    row_dict[header] = text
        data_rows.append(row_dict)

    return {
        'columns': headers,
        'data': data_rows
    }


# data_files = [f"data/drug-label-00{i}-of-0013.json" for i in range(1, 14)]
data_files = ['data/drug-label-0013-of-0013.json']

drug_info = {
    "drug_name": [],
    "adverse_effects": []
}

for file in data_files:
    with open(file, 'r') as f:
        data = json.load(f)
    
    drug_labels = data['results']

    for drug_label in drug_labels:
        drug_name = None
        if 'openfda' in drug_label and 'generic_name' in drug_label['openfda']:
            drug_name = drug_label['openfda']['generic_name'][0]
        else:
            print(f"No generic name found for {drug_label['id']}")
            continue
        assert drug_name is not None, "Drug name should not be None"
        if 'adverse_reactions_table' in drug_label:
            html_content = drug_label['adverse_reactions_table'][0]

            ae_dict = parse_adverse_reactions_table_llm(html_content)
            print(ae_dict)

            # add this to all the information
            drug_info["drug_name"].append(drug_name)
            drug_info["adverse_effects"].append(ae_dict)
        else:
            print(f"No adverse reactions table found for {drug_name}")
            continue

# turn the drug_info into a pandas dataframe
columns = set()
num_entries = len(drug_info['adverse_effects'])
for i in range(num_entries):
    for col in drug_info['adverse_effects'][i].keys():
        columns.add(col)

columns = list(columns)
columns.insert(0, 'drug_name')
df = pd.DataFrame(columns=columns)
for i in range(num_entries):
    row = [drug_info['drug_name'][i]]
    for col in columns[1:]:
        if col in drug_info['adverse_effects'][i]:
            row.append(drug_info['adverse_effects'][i][col])
        else:
            # insert NaN
            row.append(np.nan)
    df.loc[i] = row

df.to_csv('data/adverse_effects.csv', index=False)