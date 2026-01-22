import requests
import os
from dotenv import load_dotenv

load_dotenv()

def update_zotero_bib() -> None:
    """Fetches bibliography data from Zotero and saves it to the specified paper path (in the .env file)."""
    api_key = os.environ['ZOTERO_API_KEY']
    user_id = os.environ['ZOTERO_USER_ID']
    url = f"https://api.zotero.org/users/{user_id}/items"
    
    headers = {"Zotero-API-Key": api_key}
    params = {"format": "bibtex"}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        paper_path = os.environ['PAPER_PATH']
        path = os.path.join(paper_path, "refs.bib")
        bibtex_content = response.text
        with open(path, "w", encoding="utf-8") as f:
            f.write(bibtex_content)

        print(f"Saved to {path}")
    else:
        print("Error:", response.status_code, response.text)
        
if __name__ == "__main__":
    update_zotero_bib()
