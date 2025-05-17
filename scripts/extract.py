"""
Last edited: 2025-05-04
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
You are a bioinformatician that extracts the adverse effects from drug labels of FDA-approved drugs.

Your task is to extract the adverse effects from given text and HTML tables from **Section 6 (Adverse Reactions)** of a drug product label and return them in a structured format.

You will be provided with:
- The **drug name**,
- The **route of administration** (e.g., oral, intravenous),
- And a block of text containing HTML tables and descriptions.

The structured output should be in **JSON format** with the following schema:

```json
{
  "drug_name": "string",
  "drug_route": "string",
  "adverse_effects": [
    {
      "adverse_effect": "string (standardized to MedDRA, all lowercase)",
      "percentage": float or null,
      "placebo": float or null
    }
  ]
}
```

### Important Extraction Rules:

1. **Extract adverse effects and their incidence (%) from the selected treatment arm column.**

   * Also extract the corresponding value from the placebo column if available.

2. **Normalize percentage values:**

   * Convert strings like `(<1%)`, `(23 %)`, or `15 (15)` into numeric floats between 0 and 100.
   * Use the number in parentheses as the percentage.
   * If a value is `0`, return `0.0`; if value is `<1`, return a float such as `0.5`.

3. **Standardize adverse effect names using the MedDRA dictionary and lowercase formatting.**

4. If any value (either treatment or placebo) is missing, set it to `null`.

5. If adverse effects are grouped under body systems (e.g., "Nervous system disorders"), ignore these headings for extraction purposes — they are not adverse effects themselves.

6. Do not include duplicate adverse effects.
   - If the same adverse effect appears more than once (even with different values), include only the **first occurrence** based on the order in the table or text.
   - Discard subsequent instances of the same adverse effect.
   - Use string matching after standardizing the name (e.g., "Dry mouth" and "dry mouth" both map to "dry mouth").

7. Preserve the order the adverse effects appear in the table.

Return **only** the structured JSON. Do not include any commentary or explanatory text.
"""

def parse_adverse_reactions_table_llm(html_string: str) -> dict:
    """
    Parse an HTML table of adverse reactions and return structured data using OpenAI API.

    Args:
        html_string (str): String containing HTML table.
    Returns:
        {
            'drug_route': str,
            'drug_name': str,
            'adverse_effects': list of dictionaries: [
                {
                    'adverse_effect': str,
                    'percentage': float,
                    'placebo': float,
                }
            ],
            'spl-set-id': str
        }
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": instructions},
            {"role": "user", "content": html_string}
        ],
        # max_tokens=500
    )
    json_response = response.choices[0].message.content
    # strip the ```json and ``` from the start and end of the response
    json_response = json_response.strip().lstrip('```json').rstrip('```')
    # print(f"json response: {json_response}")
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
# data_files = ['data/drug-label-0013-of-0013.json', 'data/drug-label-0012-of-0013.json']
data_files = ['data/drug-label-0013-of-0013.json']

drug_info = []

cnt = 0
cnt_limit = 25

for file in data_files:
    with open(file, 'r') as f:
        data = json.load(f)
    
    drug_labels = data['results']

    if (cnt > cnt_limit):
        print(f"Reached {cnt_limit} drug labels, stopping.")
        break

    for drug_label in drug_labels:
        if (cnt > cnt_limit):
            print(f"Reached {cnt_limit} drug labels, stopping.")
            break
        drug_name = None
        if 'openfda' in drug_label and 'generic_name' in drug_label['openfda']:
            drug_name = drug_label['openfda']['generic_name'][0]
        else:
            print(f"No generic name found for {drug_label['id']}")
            continue
        assert drug_name is not None, "Drug name should not be None"
        if 'adverse_reactions_table' in drug_label and 'adverse_reactions' in drug_label and 'openfda' in drug_label and 'route' in drug_label['openfda']:
            html_content = drug_label['adverse_reactions'][0]
            for i in range(1, len(drug_label['adverse_reactions'])):
                html_content += drug_label['adverse_reactions'][i]
            for i in range(0, len(drug_label['adverse_reactions_table'])):
                html_content += drug_label['adverse_reactions_table'][i]

            ae_dict = parse_adverse_reactions_table_llm(html_content)
            ae_dict['drug_spl_set_id'] = drug_label['openfda']['spl_set_id'][0]
            ae_dict['drug_route'] = drug_label['openfda']['route'][0]
            # print(ae_dict)

            # add this to all the information
            print('added information')
            cnt += 1
            drug_info.append(ae_dict)
        else:
            print(f"No adverse reactions table found for {drug_name}")
            continue

# drump the drug_info into a JSON file
print(f"Writing {len(drug_info)} drug labels to file.")
with open('data/drug_info.json', 'w') as f:
    f.write(json.dumps(drug_info, indent=4))
print("Done.")
