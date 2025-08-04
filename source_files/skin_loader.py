# skin_loader.py
#
# MODIFIED FOR VERTEX AI:
# - Completely rewritten to read all assets directly from GCS.
# - Uses the google-cloud-storage library to list and download files.
# - Assets are loaded into in-memory buffers (io.BytesIO) for PIL to read.

import os
from PIL import Image
from utils import print_status
import configparser
import io
from google.cloud import storage
from urllib.parse import urlparse

# --- Global GCS Client ---
storage_client = None


def get_gcs_client():
    """Initializes and returns a global GCS client."""
    global storage_client
    if storage_client is None:
        storage_client = storage.Client()
    return storage_client


def parse_gcs_path(gcs_path):
    """Parses a GCS path into bucket name and blob name."""
    parsed = urlparse(gcs_path)
    if not parsed.scheme == 'gs':
        raise ValueError(f"Path must be a GCS path (gs://...). Got: {gcs_path}")
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip('/')
    return bucket_name, blob_name


def _list_blobs_in_gcs_prefix(gcs_prefix):
    """Lists all blob objects within a given GCS prefix path."""
    client = get_gcs_client()
    bucket_name, prefix = parse_gcs_path(gcs_prefix)
    if not prefix.endswith('/'):
        prefix += '/'
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    # Return a dictionary mapping the filename (e.g., 'cursor.png') to the blob object
    return {blob.name.split('/')[-1]: blob for blob in blobs if not blob.name.endswith('/')}


def _load_single_asset_gcs(asset_name, skin_blobs, default_skin_blobs):
    """
    Internal helper to load one asset from GCS blob dictionaries,
    checking for @2x versions first.
    """
    high_res_name = f'{asset_name}@2x.png'
    low_res_name = f'{asset_name}.png'

    blob_to_load = None
    is_high_res = False

    # Prioritize the main skin folder
    if high_res_name in skin_blobs:
        blob_to_load = skin_blobs[high_res_name]
        is_high_res = True
    elif low_res_name in skin_blobs:
        blob_to_load = skin_blobs[low_res_name]
    # Fallback to the default skin folder
    elif default_skin_blobs and high_res_name in default_skin_blobs:
        blob_to_load = default_skin_blobs[high_res_name]
        is_high_res = True
    elif default_skin_blobs and low_res_name in default_skin_blobs:
        blob_to_load = default_skin_blobs[low_res_name]

    if blob_to_load:
        try:
            # Download blob content into an in-memory byte stream
            content = blob_to_load.download_as_bytes()
            img = Image.open(io.BytesIO(content)).convert('RGBA')

            if is_high_res and img.width > 0 and img.height > 0:
                new_size = (max(1, img.width // 2), max(1, img.height // 2))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            print_status(f"Could not load GCS asset '{blob_to_load.name}': {e}", level="ERROR")
    return None


def load_skin_assets(skin_gcs_path):
    """
    Loads all necessary image assets from a skin folder in GCS, with a fallback
    to a 'default' skin folder if an asset is missing.
    """
    print_status(f"Loading skin assets from GCS path: {skin_gcs_path}")
    assets = {}
    combo_colors = []

    get_gcs_client()

    base_skins_dir = os.path.dirname(skin_gcs_path.rstrip('/'))
    default_skin_gcs_path = f"{base_skins_dir}/default"
    print_status(f"Using fallback GCS skin path: {default_skin_gcs_path}")

    try:
        skin_blobs = _list_blobs_in_gcs_prefix(skin_gcs_path)
        default_skin_blobs = _list_blobs_in_gcs_prefix(default_skin_gcs_path)
    except Exception as e:
        print_status(f"CRITICAL: Failed to list files in GCS for skin '{skin_gcs_path}'. Error: {e}", level="ERROR")
        return None, []

    skin_ini_blob = skin_blobs.get('skin.ini')
    config = configparser.ConfigParser(strict=False)

    score_prefix = 'score'
    combo_prefix = 'combo'
    score_overlap = 0
    combo_overlap = 0
    input_overlay_text_color = (255, 255, 255)

    if skin_ini_blob:
        try:
            ini_content = skin_ini_blob.download_as_text(encoding='utf-8-sig')

            cleaned_lines = []
            for line in ini_content.splitlines():
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    cleaned_lines.append(line)
                elif '=' in line or ':' in line:
                    cleaned_lines.append(line)
                elif line.startswith('#') or line.startswith(';'):
                    cleaned_lines.append(line)

            cleaned_content = "\n".join(cleaned_lines)
            if not cleaned_content.strip().startswith('[General]'):
                cleaned_content = '[General]\n' + cleaned_content

            config.read_string(cleaned_content)

            font_section = 'Fonts' if config.has_section('Fonts') else 'General'
            if config.has_option(font_section, 'ScorePrefix'): score_prefix = config.get(font_section, 'ScorePrefix')
            if config.has_option(font_section, 'ComboPrefix'): combo_prefix = config.get(font_section, 'ComboPrefix')
            if config.has_option(font_section, 'ScoreOverlap'): score_overlap = int(
                config.get(font_section, 'ScoreOverlap'))
            if config.has_option(font_section, 'ComboOverlap'): combo_overlap = int(
                config.get(font_section, 'ComboOverlap'))

            if config.has_section('Colours'):
                i = 1
                while True:
                    key = f'Combo{i}'
                    if config.has_option('Colours', key):
                        color_str = config.get('Colours', key)
                        try:
                            rgb = tuple(int(p.strip()) for p in color_str.split(','))
                            if len(rgb) == 3: combo_colors.append(rgb)
                        except (ValueError, IndexError):
                            print_status(f"Could not parse color for {key} in skin.ini", level="WARN")
                        i += 1
                    else:
                        break

                if config.has_option('Colours', 'InputOverlayText'):
                    color_str = config.get('Colours', 'InputOverlayText')
                    try:
                        rgb = tuple(int(p.strip()) for p in color_str.split(','))
                        if len(rgb) == 3: input_overlay_text_color = rgb
                    except Exception:
                        pass
        except Exception as e:
            skin_name = skin_gcs_path.strip('/').split('/')[-1]
            print_status(f"An error occurred while parsing skin.ini for '{skin_name}': {e}", level="ERROR")

    assets['score-overlap'] = score_overlap
    assets['combo-overlap'] = combo_overlap
    assets['input-overlay-text-color'] = input_overlay_text_color

    hitcircle_asset = _load_single_asset_gcs('hitcircle', skin_blobs, default_skin_blobs)
    hitcircleoverlay_asset = _load_single_asset_gcs('hitcircleoverlay', skin_blobs, default_skin_blobs)

    if not hitcircleoverlay_asset:
        print_status("Skin is missing hitcircleoverlay. Checking for default-digit fallback...", level="INFO")
        has_all_digits = all(f'default-{i}.png' in skin_blobs or f'default-{i}@2x.png' in skin_blobs for i in range(10))
        if has_all_digits:
            print_status("Found complete default-digit set. Using as fallback for overlay.", level="INFO")
            fallback_digit_asset = _load_single_asset_gcs('default-1', skin_blobs, default_skin_blobs)
            if fallback_digit_asset:
                hitcircleoverlay_asset = fallback_digit_asset
                hitcircle_asset = None
                assets['is_digit_fallback'] = True
        else:
            print_status("No default-digit set found. Fallback to default skin overlay.", level="WARN")
            hitcircleoverlay_asset = _load_single_asset_gcs('hitcircleoverlay', {}, default_skin_blobs)

    if not hitcircle_asset and not assets.get('is_digit_fallback'):
        hitcircle_asset = _load_single_asset_gcs('hitcircle', {}, default_skin_blobs)

    assets['hitcircle'] = hitcircle_asset
    assets['hitcircleoverlay'] = hitcircleoverlay_asset

    base_assets = [
        'approachcircle', 'cursor', 'cursortrail', 'hit0', 'hit50', 'hit100',
        'hit300', 'sliderfollowcircle', 'reversearrow', 'hitcircleselect',
        'spinner-approachcircle', 'spinner-background', 'spinner-circle',
        'inputoverlay-background', 'inputoverlay-key', 'sliderb', 'scorebar-bg', 'scorebar-colour',
        'spinner-glow', 'spinner-middle', 'spinner-top'
    ]
    for name in base_assets:
        if name not in ['hitcircle', 'hitcircleoverlay']:
            assets[name] = _load_single_asset_gcs(name, skin_blobs, default_skin_blobs)

    for i in range(10):
        assets[f'default-{i}'] = _load_single_asset_gcs(f'default-{i}', skin_blobs, default_skin_blobs)
        assets[f'score-{i}'] = _load_single_asset_gcs(f'{score_prefix}-{i}', skin_blobs, default_skin_blobs)
        assets[f'combo-{i}'] = _load_single_asset_gcs(f'{combo_prefix}-{i}', skin_blobs,
                                                      default_skin_blobs) or assets.get(f'score-{i}')

    assets['score-dot'] = _load_single_asset_gcs(f'{score_prefix}-dot', skin_blobs, default_skin_blobs)
    assets['score-percent'] = _load_single_asset_gcs(f'{score_prefix}-percent', skin_blobs, default_skin_blobs)
    assets['score-x'] = _load_single_asset_gcs(f'{score_prefix}-x', skin_blobs, default_skin_blobs)
    assets['combo-x'] = _load_single_asset_gcs(f'{combo_prefix}-x', skin_blobs, default_skin_blobs) or assets.get(
        'score-x')

    key_up_asset = _load_single_asset_gcs('inputoverlay-key', skin_blobs, default_skin_blobs)
    assets['inputoverlay-key'] = key_up_asset
    assets['inputoverlay-key-down'] = _load_single_asset_gcs('inputoverlay-key-down', skin_blobs,
                                                             default_skin_blobs) or key_up_asset

    assets['slider-tick'] = _load_single_asset_gcs('sliderscorepoint', skin_blobs,
                                                   default_skin_blobs) or _load_single_asset_gcs('slider-tick',
                                                                                                 skin_blobs,
                                                                                                 default_skin_blobs)

    if 'cursor' not in assets or assets['cursor'] is None:
        print_status("Critical asset 'cursor.png' not found even after fallbacks. Exiting.", level="ERROR")
        return None, []
    if ('hitcircleoverlay' not in assets or assets['hitcircleoverlay'] is None) and \
            ('hitcircle' not in assets or assets['hitcircle'] is None):
        print_status("Critical assets for hit circles not found even after fallbacks. Exiting.", level="ERROR")
        return None, []

    print_status(f"Successfully loaded {len(assets)} skin assets from GCS (including fallbacks).")
    if combo_colors:
        print_status(f"Found {len(combo_colors)} combo colors in skin.ini.", level="INFO")

    return assets, combo_colors