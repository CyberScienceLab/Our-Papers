import csv
import json


def csv_to_json(csv_file_path, json_file_path):
    """
    Loads a CSV file, converts it to JSON, and saves it to a JSON file.

    Args:
        csv_file_path (str): The path to the CSV file.
        json_file_path (str): The path to save the JSON file.
    """
    data = []
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)

        with open(json_file_path, 'w', encoding='utf-8') as jsonfile:
            # indent for pretty printing
            json.dump(data, jsonfile, indent=4, ensure_ascii=False)

        print(f"CSV data converted to JSON and saved to {json_file_path}")

    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


csv_to_json('/home/researchuser/LLMSec/LLMSecurity/RedHit/data/prompt_injections.csv',
            '/home/researchuser/LLMSec/LLMSecurity/RedHit/data/prompt_injections.json')
