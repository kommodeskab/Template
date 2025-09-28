import requests
import os

def update_zotero_bib(
    user_id : str,
    collection_id : str | None = None,
    api_key : str | None = None
) -> None:
    base_url = f"https://api.zotero.org/users/{user_id}"

    if api_key is None:
        assert (api_key := os.environ['ZOTERO_API_KEY']) is not None, "ZOTERO_API_KEY environment variable not set"

    if collection_id:
        url = f"{base_url}/collections/{collection_id}/items"
    else:
        url = f"{base_url}/items"

    headers = {'Zotero-API-Key': api_key}
    params = {'format': 'bibtex'}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        path = "refs.bib"
        bibtex_content = response.text
        with open(path, "w", encoding="utf-8") as f:
            f.write(bibtex_content)

        print(f"Saved to {path}")
    else:
        print("Error:", response.status_code, response.text)