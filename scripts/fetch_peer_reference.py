"""Retrieve the pinned public notebook for read-only comparison provenance."""
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]
COMMIT='008ebf84c66cb0b92e357d8212a0d2c2333022bd'
URL=f'https://raw.githubusercontent.com/kdw123654/digital-project/{COMMIT}/my_model.ipynb'

def main():
    target=ROOT/'data/reference/peer'
    target.mkdir(parents=True,exist_ok=True)
    with urlopen(Request(URL,headers={'User-Agent':'digital-health-research-reproduction'}),timeout=30) as response:
        content=response.read()
    notebook=json.loads(content)
    if not isinstance(notebook.get('cells'),list):raise ValueError('Not a valid notebook')
    (target/'my_model.ipynb').write_bytes(content)
    (target/'commit.json').write_text(json.dumps({'sha':COMMIT},indent=2),encoding='utf-8')
    print('Saved pinned reference for read-only inspection; notebook was not executed.')

if __name__=='__main__':main()
