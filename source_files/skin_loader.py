# skin_loader.py
#
# Contains the logic for loading image assets from a skin folder.
# MODIFIED: Added an 'is_digit_fallback' flag to the assets dictionary when the
#           digit-based fallback for hitcircleoverlay is successfully used.

import os
from PIL import Image
from utils import print_status
import configparser
import io


def _load_single_asset(base_path, asset_name):
    """
    Internal helper to load one asset, checking for @2x versions first.
    Assets are loaded as-is, without being cropped.
    """
    if not asset_name or not base_path: return None

    high_res_path = os.path.join(base_path, f'{asset_name}@2x.png')
    low_res_path = os.path.join(base_path, f'{asset_name}.png')

    loaded_path = None
    is_high_res = False

    if os.path.exists(high_res_path):
        loaded_path = high_res_path
        is_high_res = True
    elif os.path.exists(low_res_path):
        loaded_path = low_res_path

    if loaded_path:
        try:
            img = Image.open(loaded_path).convert('RGBA')

            if is_high_res and img.width > 0 and img.height > 0:
                new_size = (max(1, img.width // 2), max(1, img.height // 2))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            print_status(f"Could not load asset '{os.path.basename(loaded_path)}': {e}", level="ERROR")
    return None


def load_asset_with_fallback(primary_path, fallback_path, asset_name):
    """
    Loads an asset, trying the primary path and then the fallback path.
    """
    asset = _load_single_asset(primary_path, asset_name)
    if asset:
        return asset
    return _load_single_asset(fallback_path, asset_name)


def load_skin_assets(skin_path):
    """
    Loads all necessary image assets from the skin folder, with a fallback
    to a 'default' skin folder if an asset is missing.
    """
    print_status(f"Loading skin assets from: {skin_path}")
    assets = {}
    combo_colors = []

    default_skin_path = os.path.join(os.path.dirname(skin_path), 'default')
    print_status(f"Using fallback skin path: {default_skin_path}")

    skin_ini_path = os.path.join(skin_path, 'skin.ini')
    config = configparser.ConfigParser(strict=False)

    scorebar_offset = (5, 16)
    score_prefix = 'score'
    combo_prefix = 'combo'
    score_overlap = 0
    combo_overlap = 0
    input_overlay_text_color = (255, 255, 255)

    if os.path.exists(skin_ini_path):
        try:
            cleaned_lines = []
            with open(skin_ini_path, 'r', encoding='utf-8-sig') as f:
                for line in f:
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
                            print_status(f"Could not parse color for {key} in {skin_ini_path}", level="WARN")
                        i += 1
                    else:
                        break

                if config.has_option('Colours', 'ScorebarColour'):
                    offset_str = config.get('Colours', 'ScorebarColour')
                    try:
                        parts = [int(p.strip()) for p in offset_str.split(',')]
                        if len(parts) == 2: scorebar_offset = tuple(parts)
                    except Exception:
                        pass
                if config.has_option('Colours', 'InputOverlayText'):
                    color_str = config.get('Colours', 'InputOverlayText')
                    try:
                        rgb = tuple(int(p.strip()) for p in color_str.split(','))
                        if len(rgb) == 3: input_overlay_text_color = rgb
                    except Exception:
                        pass
        except configparser.Error as e:
            print_status(f"An error occurred while parsing skin.ini for '{os.path.basename(skin_path)}': {e}",
                         level="ERROR")

    assets['scorebar-offset'] = scorebar_offset
    assets['score-overlap'] = score_overlap
    assets['combo-overlap'] = combo_overlap
    assets['input-overlay-text-color'] = input_overlay_text_color

    if os.path.exists(os.path.join(skin_path, 'scorebar-bg.png')) and \
            os.path.exists(os.path.join(skin_path, 'scorebar-colour.png')):
        assets['scorebar-bg'] = _load_single_asset(skin_path, 'scorebar-bg')
        assets['scorebar-colour'] = _load_single_asset(skin_path, 'scorebar-colour')
    else:
        assets['scorebar-bg'] = _load_single_asset(default_skin_path, 'scorebar-bg')
        assets['scorebar-colour'] = _load_single_asset(default_skin_path, 'scorebar-colour')

    hitcircle_asset = _load_single_asset(skin_path, 'hitcircle')
    hitcircleoverlay_asset = _load_single_asset(skin_path, 'hitcircleoverlay')

    if not hitcircleoverlay_asset:
        print_status("Skin is missing hitcircleoverlay. Checking for default-digit fallback...", level="INFO")
        has_all_digits = all(_load_single_asset(skin_path, f'default-{i}') for i in range(10))
        if has_all_digits:
            print_status("Found complete default-digit set. Using as fallback for overlay.", level="INFO")
            fallback_digit_asset = _load_single_asset(skin_path, 'default-1')
            if fallback_digit_asset:
                hitcircleoverlay_asset = fallback_digit_asset
                hitcircle_asset = None
                # --- MODIFICATION: Add the flag ---
                assets['is_digit_fallback'] = True
        else:
             print_status("No default-digit set found in skin. Proceeding to global default.", level="WARN")

    if not hitcircle_asset:
        hitcircle_asset = _load_single_asset(default_skin_path, 'hitcircle')
    if not hitcircleoverlay_asset:
        hitcircleoverlay_asset = _load_single_asset(default_skin_path, 'hitcircleoverlay')

    assets['hitcircle'] = hitcircle_asset
    assets['hitcircleoverlay'] = hitcircleoverlay_asset

    base_assets = [
        'approachcircle', 'cursor', 'cursortrail', 'hit0', 'hit50', 'hit100',
        'hit300', 'sliderfollowcircle', 'reversearrow', 'hitcircleselect',
        'spinner-approachcircle', 'spinner-background', 'spinner-circle',
        'inputoverlay-background', 'inputoverlay-key', 'sliderb'
    ]
    spinner_layers = ['spinner-glow', 'spinner-middle', 'spinner-top']
    base_assets.extend(spinner_layers)

    for name in base_assets:
        if name not in ['hitcircle', 'hitcircleoverlay']:
            asset = load_asset_with_fallback(skin_path, default_skin_path, name)
            if asset: assets[name] = asset

    for i in range(10):
        assets[f'default-{i}'] = load_asset_with_fallback(skin_path, default_skin_path, f'default-{i}')
        assets[f'score-{i}'] = load_asset_with_fallback(skin_path, default_skin_path, f'{score_prefix}-{i}')
        assets[f'combo-{i}'] = load_asset_with_fallback(skin_path, default_skin_path,
                                                        f'{combo_prefix}-{i}') or assets.get(f'score-{i}')

    assets['score-dot'] = load_asset_with_fallback(skin_path, default_skin_path, f'{score_prefix}-dot')
    assets['score-percent'] = load_asset_with_fallback(skin_path, default_skin_path, f'{score_prefix}-percent')
    assets['score-x'] = load_asset_with_fallback(skin_path, default_skin_path, f'{score_prefix}-x')
    assets['combo-x'] = load_asset_with_fallback(skin_path, default_skin_path, f'{combo_prefix}-x') or assets.get(
        'score-x')

    key_up_asset = load_asset_with_fallback(skin_path, default_skin_path, 'inputoverlay-key')
    assets['inputoverlay-key'] = key_up_asset
    assets['inputoverlay-key-down'] = load_asset_with_fallback(skin_path, default_skin_path,
                                                               'inputoverlay-key-down') or key_up_asset

    assets['slider-tick'] = load_asset_with_fallback(skin_path, default_skin_path,
                                                     'sliderscorepoint') or load_asset_with_fallback(skin_path,
                                                                                                     default_skin_path,
                                                                                                     'slider-tick')

    if 'cursor' not in assets:
        print_status("Critical asset 'cursor.png' not found even after fallbacks. Exiting.", level="ERROR")
        return None, []
    if 'hitcircleoverlay' not in assets and 'hitcircle' not in assets:
        print_status("Critical assets for hit circles not found even after fallbacks. Exiting.", level="ERROR")
        return None, []


    print_status(f"Successfully loaded {len(assets)} skin assets (including fallbacks).")
    if combo_colors:
        print_status(f"Found {len(combo_colors)} combo colors in skin.ini.", level="INFO")

    return assets, combo_colors