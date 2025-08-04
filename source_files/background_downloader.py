import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_image(beatmap_id, save_folder):
    url = f"https://assets.ppy.sh/beatmaps/{beatmap_id}/covers/raw.jpg"
    file_path = os.path.join(save_folder, f"{beatmap_id}.jpg")

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return f"✅ Downloaded {beatmap_id}"
        else:
            return f"❌ Not found (ID {beatmap_id})"
    except Exception as e:
        return f"⚠️ Error with ID {beatmap_id}: {e}"

def download_beatmap_covers_fast(start_id, end_id, save_folder="downloaded_covers", max_workers=20):
    os.makedirs(save_folder, exist_ok=True)
    beatmap_ids = range(start_id, end_id + 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image, bid, save_folder) for bid in beatmap_ids]
        for future in as_completed(futures):
            print(future.result())

# Example usage
download_beatmap_covers_fast(start_id=2291150, end_id=2391250, max_workers=50)
