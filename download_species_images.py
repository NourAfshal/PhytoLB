"""
download_species_images.py
Run once from your project root. Already-downloaded images are skipped.
Usage: python download_species_images.py
"""

import os, time, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUTPUT_DIR = os.path.join("static", "species")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIKI_TITLES = {
    "Abies cilicica":                   "Abies cilicica",
    "Acer hyrcanum tauricolum":         "Acer hyrcanum",
    "Acer monspessulanum microphyllum": "Acer monspessulanum",
    "Acer obtusifolium":                "Acer obtusifolium",
    "Amygdalus orientalis":             "Prunus orientalis",
    "Arbutus andrachne":                "Arbutus andrachne",
    "Cedrus libani":                    "Cedrus libani",
    "Ceratonia siliqua":                "Ceratonia siliqua",
    "Cercis siliquastrum":              "Cercis siliquastrum",
    "Crataegus azarolus":               "Crataegus azarolus",
    "Cupressus sempervirens":           "Cupressus sempervirens",
    "Fraxinus ornus":                   "Fraxinus ornus",
    "Juniperus drupacea":               "Juniperus drupacea",
    "Juniperus excelsa":                "Juniperus excelsa",
    "Laurus nobilis":                   "Laurus nobilis",
    "Malus trilobata":                  "Malus trilobata",
    "Pinus pinea":                      "Pinus pinea",
    "Pyrus syriaca":                    "Pyrus syriaca",
    "Quercus cedrorum":                 "Quercus cerris",
    "Quercus cerris":                   "Quercus cerris",
    "Quercus coccifera calliprinos":    "Quercus coccifera",
    "Quercus infectoria boissieri":     "Quercus infectoria",
    "Quercus ithaburensis":             "Quercus ithaburensis",
    "Quercus kotschyana":               "Quercus look",
    "Quercus look":                     "Quercus look",
    "Sorbus torminalis":                "Sorbus torminalis",
    "Styrax officinalis":               "Styrax officinalis",
}

HEADERS    = {"User-Agent": "PhytoLB/1.0 (educational plant-suitability tool)"}
API_DELAY  = 4.0   # seconds between Wikipedia API calls
IMG_DELAY  = 3.0   # seconds between image downloads
RETRY_WAIT = 20.0  # seconds to wait after a 429
MAX_RETRY  = 3

def make_session():
    s = requests.Session()
    s.headers.update(HEADERS)
    adapter = HTTPAdapter(max_retries=Retry(total=2, backoff_factor=2,
                          status_forcelist=[500, 502, 503, 504]))
    s.mount("https://", adapter)
    return s

SESSION = make_session()

def safe_filename(key):
    return key.lower().replace(" ", "_").replace("/", "_") + ".jpg"

def fetch_thumb(wiki_title):
    params = {"action":"query","titles":wiki_title,"prop":"pageimages",
              "format":"json","pithumbsize":500,"pilicense":"any"}
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = SESSION.get("https://en.wikipedia.org/w/api.php", params=params, timeout=12)
            if r.status_code == 429:
                wait = RETRY_WAIT * attempt
                print(f"  ⏳ API rate limited — waiting {wait:.0f}s (attempt {attempt}/{MAX_RETRY})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            for page in r.json().get("query",{}).get("pages",{}).values():
                thumb = page.get("thumbnail",{}).get("source")
                if thumb:
                    return thumb
            return None
        except Exception as e:
            print(f"  API error: {e}")
            return None
    return None

def download(url, path):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = SESSION.get(url, timeout=20, stream=True)
            if r.status_code == 429:
                wait = RETRY_WAIT * attempt
                print(f"  ⏳ Download rate limited — waiting {wait:.0f}s (attempt {attempt}/{MAX_RETRY})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            print(f"  ✓ saved ({os.path.getsize(path)//1024} KB)")
            return True
        except Exception as e:
            print(f"  ✗ {e}")
            if os.path.exists(path): os.remove(path)
            return False
    return False

def main():
    total = len(WIKI_TITLES)
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Delays: {API_DELAY}s API / {IMG_DELAY}s image  (be patient, ~{total*7//60+1} min)\n")
    ok, results = 0, {}

    for i, (key, wiki) in enumerate(WIKI_TITLES.items(), 1):
        fname = safe_filename(key)
        fpath = os.path.join(OUTPUT_DIR, fname)

        if os.path.exists(fpath) and os.path.getsize(fpath) > 5000:
            print(f"[{i:2}/{total}] SKIP  {key}")
            results[key] = True; ok += 1; continue

        print(f"[{i:2}/{total}] FETCH {key}")
        thumb = fetch_thumb(wiki)
        time.sleep(API_DELAY)

        if thumb:
            print(f"  → {thumb}")
            if download(thumb, fpath):
                results[key] = True; ok += 1
            else:
                results[key] = False
            time.sleep(IMG_DELAY)
        else:
            print("  ✗ no Wikipedia thumbnail")
            results[key] = False

    print(f"\n── Done: {ok}/{total} images ──")
    missing = [k for k,v in results.items() if not v]
    if missing:
        print("Still missing (letter placeholder will show):")
        for m in missing: print(f"  • {m}")

if __name__ == "__main__":
    main()