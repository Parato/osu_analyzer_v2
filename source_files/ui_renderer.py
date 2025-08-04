# ui_renderer.py
#
# Contains the logic for rendering the game's user interface elements,
# such as the HP bar, score, and combo.
# FINAL FIX 27: Corrected the key overlay's rotation direction to be clockwise (-90).
#               Integrated the skin.ini's InputOverlayText color for the key counters.
# MODIFIED: Stretched the input overlay background by 5%.
# MODIFIED: Added press-and-shrink animation and conditional tinting to keys.
# MODIFIED: Aspect ratio of key assets is now calculated based on width.
# MODIFIED: Changed variable name from combo_image to combo_img for consistency.

from PIL import Image, ImageDraw, ImageFont
import config_generator as cfg
import math


def tint_image(image, color):
    """
    Tints an image by using its alpha channel as a mask for a solid color.
    """
    if not image or not color:
        return image
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    if len(color) == 3:
        color_rgba = (*color, 255)
    else:
        color_rgba = color
    color_image = Image.new("RGBA", image.size, color_rgba)
    # The original image's alpha channel is used as the mask
    final_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    final_image.paste(color_image, (0, 0), mask=image)
    return final_image


def render_hp_bar(assets, current_hp):
    """
    Renders the HP bar, correctly clamping its width between a minimum and
    maximum value while maintaining the asset's aspect ratio.
    """
    hp_bar_bg = assets.get('scorebar-bg')
    hp_bar_colour = assets.get('scorebar-colour')
    scorebar_offset = assets.get('scorebar-offset', (5, 16))

    if not hp_bar_bg or not hp_bar_colour:
        return None

    # Clamp the target width so it's never larger than the max or smaller than the min.
    calculated_width = int(cfg.OUTPUT_RESOLUTION[0] * 0.55)
    target_bg_width = max(cfg.UI_HP_BAR_MIN_WIDTH_PX, min(calculated_width, cfg.UI_HP_BAR_MAX_WIDTH_PX))

    original_bg_width, original_bg_height = hp_bar_bg.size
    if original_bg_width == 0:
        return Image.new('RGBA', (1, 1))

    # Calculate scale ratio based on the final, clamped width.
    scale_ratio = target_bg_width / original_bg_width

    target_bg_height = max(1, int(original_bg_height * scale_ratio))
    target_bg_size = (target_bg_width, target_bg_height)

    resized_bg = hp_bar_bg.resize(target_bg_size, Image.Resampling.LANCZOS)

    new_colour_size = (max(1, int(hp_bar_colour.width * scale_ratio)), max(1, int(hp_bar_colour.height * scale_ratio)))
    resized_colour = hp_bar_colour.resize(new_colour_size, Image.Resampling.LANCZOS)

    final_hp_bar = resized_bg.copy()

    final_color_width = int(resized_colour.width * current_hp)
    if final_color_width > 0:
        cropped_colour = resized_colour.crop((0, 0, final_color_width, resized_colour.height))
        paste_x = int(scorebar_offset[0] * scale_ratio)
        paste_y = int(scorebar_offset[1] * scale_ratio)
        final_hp_bar.paste(cropped_colour, (paste_x, paste_y), cropped_colour)

    return final_hp_bar


def render_score_combo(assets, score, combo):
    score_overlap = assets.get('score-overlap', 0)
    combo_overlap = assets.get('combo-overlap', 0)
    scaled_score_overlap = int(score_overlap * cfg.UI_SCALE)
    scaled_combo_overlap = int(combo_overlap * cfg.UI_SCALE)

    score_image = None
    score_str = str(score).zfill(8)
    score_digit_assets = [assets.get(f'score-{d}_ui') for d in score_str]
    if not any(d is None for d in score_digit_assets):
        total_width = sum(d.width for d in score_digit_assets) - (scaled_score_overlap * (len(score_digit_assets) - 1))
        max_height = max(d.height for d in score_digit_assets)
        score_image = Image.new('RGBA', (max(0, total_width), max(0, max_height)), (0, 0, 0, 0))
        x_offset = 0
        for digit_img in score_digit_assets:
            score_image.paste(digit_img, (x_offset, 0), digit_img)
            x_offset += digit_img.width - scaled_score_overlap

    combo_img = None
    if combo > 0:
        combo_str = str(combo)
        combo_digit_assets = [assets.get(f'combo-{d}_ui') for d in combo_str]
        combo_x_asset = assets.get('combo-x_ui')
        if not any(d is None for d in combo_digit_assets) and combo_x_asset:
            all_combo_assets = combo_digit_assets + [combo_x_asset]
            total_width = sum(d.width for d in all_combo_assets) - (scaled_combo_overlap * (len(all_combo_assets) - 1))
            max_height = max(d.height for d in all_combo_assets)
            combo_img = Image.new('RGBA', (max(0, total_width), max(0, max_height)), (0, 0, 0, 0))
            x_offset = 0
            for i, digit_img in enumerate(all_combo_assets):
                combo_img.paste(digit_img, (x_offset, 0), digit_img)
                if i < len(all_combo_assets) - 1:
                    x_offset += digit_img.width - scaled_combo_overlap
    return score_image, combo_img


def render_accuracy(assets, accuracy):
    acc_str = f"{accuracy:.2f}"
    acc_images = []
    char_map = {'.': 'score-dot_ui', '%': 'score-percent_ui'}
    for char in acc_str:
        if char.isdigit():
            acc_images.append(assets.get(f'score-{char}_ui'))
        elif char in char_map:
            acc_images.append(assets.get(char_map[char]))
    acc_images.append(assets.get('score-percent_ui'))

    if any(img is None for img in acc_images): return None

    total_width = sum(img.width for img in acc_images)
    max_height = max(img.height for img in acc_images)
    accuracy_image = Image.new('RGBA', (total_width, max_height), (0, 0, 0, 0))
    x_offset = 0
    for img in acc_images:
        y_offset = max_height - img.height
        accuracy_image.paste(img, (x_offset, y_offset), img)
        x_offset += img.width
    return accuracy_image


def render_key_overlay(assets, key_state, key_press_counts):
    """
    Renders the key overlay with animated, tinted keys and a stretched background.
    """
    bg_img = assets.get('inputoverlay-background')
    key_img = assets.get('inputoverlay-key')  # We only need the base key now
    text_color = assets.get('input-overlay-text-color', (255, 255, 255))
    if not bg_img or not key_img: return None

    # --- Step 1: Define Constants & Master Scale ---
    ORIGINAL_KEY_CENTERS = {'k1': (30, 24), 'k2': (78, 24), 'm1': (126, 24), 'm2': (172, 24)}
    key_names = ['k1', 'k2', 'm1', 'm2']

    # Stretch background height by 5%
    final_overlay_height = int(cfg.OUTPUT_RESOLUTION[1] * cfg.UI_OVERLAY_HEIGHT_RATIO * 1.05)
    original_bg_width = bg_img.width
    if original_bg_width == 0: return None

    bg_scale_ratio = final_overlay_height / original_bg_width

    # --- Step 2: Create the Final Vertical Canvas ---
    scaled_bg_size = (max(1, int(bg_img.width * bg_scale_ratio)), max(1, int(bg_img.height * bg_scale_ratio)))
    horizontal_bg = bg_img.resize(scaled_bg_size, Image.Resampling.LANCZOS)
    final_canvas = horizontal_bg.rotate(-90, expand=True)

    # --- Step 3: Define Tinting Colors and Key Properties ---
    TINT_YELLOW = (255, 255, 0)
    TINT_PURPLE = (200, 50, 255)
    TINT_WHITE = (255, 255, 255)
    SHRINK_FACTOR = 0.95

    # --- Step 4: Paste Animated Keys onto the Final Vertical Canvas ---
    key_centers_on_final_canvas = {}
    key_target_height = 0 # Initialize for font scaling
    for key_name in key_names:
        is_pressed = (key_name == 'm1' and key_state.get('m1', False) and not key_state.get('k1', False)) or \
                     (key_name == 'm2' and key_state.get('m2', False) and not key_state.get('k2', False)) or \
                     (key_name in ['k1', 'k2'] and key_state.get(key_name, False))

        # Determine tint and scale based on press state
        scale_factor = SHRINK_FACTOR if is_pressed else 1.0
        if is_pressed:
            tint_color = TINT_YELLOW if key_name in ['k1', 'k2'] else TINT_PURPLE
        else:
            tint_color = TINT_WHITE

        # Tint the original key image
        tinted_key = tint_image(key_img, tint_color)

        # --- MODIFIED: Scale key asset based on the final canvas's WIDTH ---
        # This ensures the key's aspect ratio is preserved while fitting the width.
        key_target_width = final_canvas.width
        key_scale_ratio = key_target_width / tinted_key.width if tinted_key.width > 0 else 0
        key_target_height = max(1, int(tinted_key.height * key_scale_ratio))

        # Apply press-and-shrink effect
        final_key_width = int(key_target_width * scale_factor)
        final_key_height = int(key_target_height * scale_factor)


        if final_key_width <= 0 or final_key_height <= 0: continue

        key_final_scaled = tinted_key.resize((final_key_width, final_key_height), Image.Resampling.LANCZOS)

        # The horizontal position on the original bg becomes the vertical position on the final canvas
        center_y = int(ORIGINAL_KEY_CENTERS[key_name][0] * bg_scale_ratio)
        # Center the key horizontally on the final canvas
        center_x = final_canvas.width // 2

        # Paste using the calculated center to handle size changes from shrinking
        paste_x = center_x - key_final_scaled.width // 2
        paste_y = center_y - key_final_scaled.height // 2
        final_canvas.paste(key_final_scaled, (paste_x, paste_y), key_final_scaled)

        key_centers_on_final_canvas[key_name] = (center_x, center_y)

    # --- Step 5: Draw Upright Text onto the Final Canvas ---
    draw = ImageDraw.Draw(final_canvas)
    try:
        # Scale font based on the un-shrunk key's height for consistency
        font_size = int(key_target_height * 0.4)
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for key_name in key_names:
        count = str(key_press_counts.get(f'{key_name}_presses', 0))
        cx, cy = key_centers_on_final_canvas[key_name]
        # Apply the correct text color from skin.ini
        draw.text((cx, cy), count, font=font, fill=(*text_color, 220), anchor="mm")

    return final_canvas