# renderer.py
#
# Contains the core rendering logic for creating gameplay frames.
# MODIFIED: Annotation for all assets (hitcircles, sliders, misses) now uses
#           pre-calculated opaque dimensions for maximum tightness.

from PIL import Image, ImageDraw, ImageOps, ImageFont, ImageChops
import config_generator as cfg
import math
import numpy as np
import bezier

# --- Local Imports ---
import ui_renderer
from follow_points import render_follow_points

# --- Constants ---
UI_SCALE = 0.75
CURSOR_TRAIL_LENGTH = 12

# --- State ---
cursor_trail_history = []
displayed_hp = 1.0


def reset_renderer_state():
    """Resets any stateful variables in the renderer, like the cursor trail and HP bar."""
    global cursor_trail_history, displayed_hp
    cursor_trail_history = []
    displayed_hp = 1.0


# --- Helper Functions ---

def get_ar_ms(ar):
    """Converts Approach Rate (AR) to milliseconds."""
    if ar < 5:
        return 1200 + 600 * (5 - ar) / 5
    elif ar == 5:
        return 1200
    else:
        return 1200 - 750 * (ar - 5) / 5


def get_cs_radius(cs):
    """Converts Circle Size (CS) to osu!pixels radius."""
    return (109 - 9 * cs) / 2


def get_timing_windows(od):
    """Calculates the hit windows in milliseconds for 300, 100, and 50 hits."""
    return {
        '300': 80 - 6 * od,
        '100': 140 - 8 * od,
        '50': 200 - 10 * od
    }


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
    final_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    final_image.paste(color_image, (0, 0), mask=image)
    return final_image


# --- Rendering Components ---
def render_combo_number(canvas, assets, combo_number, circle_pixel_radius, main_font):
    NUMBER_HEIGHT_SCALE = 0.8
    combo_str = str(combo_number)
    digit_to_render_str = ""
    if len(combo_str) == 1:
        digit_to_render_str = combo_str
    else:
        digit_to_render_str = combo_str[-2]

    has_full_default_set = all(f'default-{i}' in assets for i in range(10))
    digit_image = assets.get(f'default-{digit_to_render_str}')

    if has_full_default_set and digit_image:
        orig_w, orig_h = digit_image.size
        if orig_h > 0:
            hit_circle_diameter = circle_pixel_radius * 2
            target_height = int(hit_circle_diameter * NUMBER_HEIGHT_SCALE)
            scale_ratio = target_height / orig_h
            target_width = int(orig_w * scale_ratio)

            if target_width > 0 and target_height > 0:
                scaled_digit = digit_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
                paste_x = (canvas.width - scaled_digit.width) // 2
                paste_y = (canvas.height - scaled_digit.height) // 2
                canvas.paste(scaled_digit, (paste_x, paste_y), scaled_digit)
                return

    draw = ImageDraw.Draw(canvas)
    font_size = int(circle_pixel_radius * 1.2)
    try:
        font = main_font.font_variant(size=font_size) if hasattr(main_font, 'font_variant') else main_font
    except IOError:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), digit_to_render_str, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (canvas.width - text_width) // 2
    text_y = (canvas.height - text_height) // 2
    draw.text((text_x, text_y), digit_to_render_str, font=font, fill=(255, 255, 255, 255))


def render_hittable_object(frame_image, assets, obj, difficulty, combo_colors, raw_annotations, is_hd, main_font,
                           use_scaling_animation=False):
    ar_ms = get_ar_ms(difficulty.get('ApproachRate', 9))
    od = difficulty.get('OverallDifficulty', 9)
    timing_windows = get_timing_windows(od)
    current_time_ms = obj['current_time_ms']
    opacity = 1.0
    scale = 1.0
    if is_hd:
        fade_in_duration = ar_ms * cfg.HD_FADE_IN_DURATION_MULTIPLIER
        fade_in_start_time = obj['time'] - ar_ms
        fade_in_end_time = fade_in_start_time + fade_in_duration
        fade_out_duration = ar_ms - fade_in_duration
        if current_time_ms < fade_in_start_time:
            opacity = 0.0
        elif current_time_ms < fade_in_end_time:
            raw_opacity = (current_time_ms - fade_in_start_time) / fade_in_duration
            opacity = raw_opacity * 0.95 + 0.05
        elif current_time_ms < obj['time']:
            raw_opacity = (obj['time'] - current_time_ms) / fade_out_duration if fade_out_duration > 0 else 0.0
            opacity = raw_opacity * 0.95 + 0.05
        else:
            opacity = 0.0
    else:
        fade_in_start_time = obj['time'] - ar_ms
        if current_time_ms < obj['time']:
            if current_time_ms < fade_in_start_time:
                opacity = 0.0
            else:
                raw_opacity = (current_time_ms - fade_in_start_time) / ar_ms
                opacity = raw_opacity * 0.95 + 0.05
        elif 'hit_time' in obj:
            time_since_hit = current_time_ms - obj['hit_time']
            if 0 <= time_since_hit < cfg.FADE_OUT_DURATION_MS:
                progress = time_since_hit / cfg.FADE_OUT_DURATION_MS
                raw_opacity = 1.0 - progress
                opacity = raw_opacity * 0.95 + 0.05
                if use_scaling_animation:
                    scale = 1.0 + 0.4 * progress
            elif time_since_hit >= cfg.FADE_OUT_DURATION_MS:
                opacity = 0.0
        else:
            miss_time = obj.get('miss_time', obj['time'] + timing_windows['50'])
            if current_time_ms > miss_time:
                time_since_miss = current_time_ms - miss_time
                if time_since_miss < cfg.FADE_OUT_DURATION_MS:
                    raw_opacity = 1.0 - (time_since_miss / cfg.FADE_OUT_DURATION_MS)
                    opacity = raw_opacity * 0.95 + 0.05
                else:
                    opacity = 0.0
    if opacity <= 0: return
    playfield_scale = cfg.PLAYFIELD_HEIGHT / 384.0
    circle_pixel_radius = int(get_cs_radius(difficulty.get('CircleSize', 4)) * playfield_scale)
    render_x = obj.get('render_x', obj['x'])
    render_y = obj.get('render_y', obj['y'])
    screen_x = int(render_x * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X)
    screen_y = int(render_y * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)

    overlay_img = assets.get('hitcircleoverlay')
    hitcircle_img = assets.get('hitcircle')
    base_size_asset = overlay_img if overlay_img else hitcircle_img
    if not base_size_asset: return
    hc_size = base_size_asset.size

    final_object_canvas = Image.new('RGBA', hc_size, (0, 0, 0, 0))
    tint_color = combo_colors[obj['combo_color_index']] if combo_colors else None
    is_digit_fallback = assets.get('is_digit_fallback', False)

    if is_digit_fallback:
        # In this case, the overlay *is* the full object. No tint, no base circle.
        if overlay_img:
            final_object_canvas.paste(overlay_img, (0, 0), overlay_img)
    else:
        # Normal rendering: tinted base circle with an overlay on top.
        if hitcircle_img:
            tinted_hitcircle = tint_image(hitcircle_img, tint_color)
            final_object_canvas = Image.alpha_composite(final_object_canvas, tinted_hitcircle)
        if overlay_img:
            final_object_canvas = Image.alpha_composite(final_object_canvas, overlay_img)

    render_combo_number(final_object_canvas, assets, obj['combo_number'], circle_pixel_radius, main_font)

    final_size = hc_size
    if scale != 1.0:
        final_size = (int(hc_size[0] * scale), int(hc_size[1] * scale))
        if final_size[0] > 0 and final_size[1] > 0:
            final_object_canvas = final_object_canvas.resize(final_size, Image.Resampling.LANCZOS)
        else:
            return

    if opacity < 1.0:
        alpha = final_object_canvas.getchannel('A')
        final_object_canvas.putalpha(alpha.point(lambda p: int(p * opacity)))

    top_left_x_paste = screen_x - final_size[0] // 2
    top_left_y_paste = screen_y - final_size[1] // 2
    frame_image.paste(final_object_canvas, (top_left_x_paste, top_left_y_paste), final_object_canvas)

    # --- FIX: Use pre-calculated opaque dimensions for annotation ---
    opaque_dims = assets.get('hitcircle_opaque_dims')
    if opaque_dims and opaque_dims[0] is not None:
        # Scale the opaque box by the animation scale
        box_w, box_h = int(opaque_dims[0] * scale), int(opaque_dims[1] * scale)
    else:
        # Fallback to full size if opaque dims aren't available
        box_w, box_h = final_size

    top_left_x_anno = screen_x - box_w // 2
    top_left_y_anno = screen_y - box_h // 2
    raw_annotations.append(
        {'class_id': cfg.CLASS_MAP['hit_circle'], 'box': [top_left_x_anno, top_left_y_anno, box_w, box_h]})
    # --- END FIX ---

    approachcircle_img = assets.get('approachcircle')
    if not is_hd and approachcircle_img and current_time_ms < obj['time']:
        time_diff = obj['time'] - current_time_ms
        ac_progress = time_diff / ar_ms
        ac_scale = 1 + 3 * ac_progress
        ac_size_w = int(hc_size[0] * ac_scale)
        ac_size_h = int(hc_size[1] * ac_scale)
        if ac_size_w > 0 and ac_size_h > 0:
            ac_resized = approachcircle_img.resize((ac_size_w, ac_size_h), Image.Resampling.LANCZOS)
            tinted_ac = tint_image(ac_resized, tint_color)
            if opacity < 1.0:
                ac_alpha = tinted_ac.getchannel('A')
                tinted_ac.putalpha(ac_alpha.point(lambda p: int(p * opacity)))
            ac_top_left_x = screen_x - ac_size_w // 2
            ac_top_left_y = screen_y - ac_size_h // 2
            frame_image.paste(tinted_ac, (ac_top_left_x, ac_top_left_y), tinted_ac)


def render_slider_ticks(frame_image, assets, obj, circle_pixel_radius, curve, opacity_multiplier, raw_annotations):
    tick_resized = assets.get('slider-tick')
    if not tick_resized: return
    num_ticks = math.ceil(obj['slides'] * obj['length'] / 100)
    if num_ticks <= 1: return

    final_tick_img = tick_resized
    if opacity_multiplier < 1.0:
        alpha = tick_resized.getchannel('A')
        final_tick_img = tick_resized.copy()
        final_tick_img.putalpha(alpha.point(lambda p: int(p * opacity_multiplier)))

    tick_size = final_tick_img.size
    for i in range(1, num_ticks):
        progress = i / num_ticks
        pos = curve.evaluate(progress)
        screen_x = int(pos[0, 0] * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X)
        screen_y = int(pos[1, 0] * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)
        top_left_x = screen_x - tick_size[0] // 2
        top_left_y = screen_y - tick_size[1] // 2

        frame_image.paste(final_tick_img, (top_left_x, top_left_y), final_tick_img)


def render_slider_follow_area(frame_image, assets, center_x, center_y, circle_pixel_radius, raw_annotations):
    follow_resized = assets.get('sliderfollowcircle')
    if not follow_resized: return

    follow_diameter_px = follow_resized.width
    top_left_x = center_x - follow_diameter_px // 2
    top_left_y = center_y - follow_diameter_px // 2
    frame_image.paste(follow_resized, (top_left_x, top_left_y), follow_resized)


def render_slider_ball(frame_image, assets, obj, combo_colors, slider_duration, circle_pixel_radius, curve,
                       raw_annotations, is_tracked, main_font):
    current_time_ms = obj['current_time_ms']
    time_since_hit = current_time_ms - obj['time']
    if slider_duration == 0: return
    progress = time_since_hit / slider_duration
    slide_progress = progress * obj['slides']
    current_slide = math.floor(slide_progress)
    progress_this_slide = slide_progress - current_slide
    if current_slide % 2 != 0: progress_this_slide = 1 - progress_this_slide

    if 0 <= progress_this_slide <= 1:
        ball_pos = curve.evaluate(progress_this_slide)
        ball_x, ball_y = ball_pos[0, 0], ball_pos[1, 0]
        screen_x = int(ball_x * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X)
        screen_y = int(ball_y * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)

        if is_tracked:
            render_slider_follow_area(frame_image, assets, screen_x, screen_y, circle_pixel_radius, raw_annotations)

        overlay_img = assets.get('hitcircleoverlay')
        hitcircle_img = assets.get('hitcircle')
        base_size_asset = overlay_img if overlay_img else hitcircle_img
        if not base_size_asset: return
        hc_size = base_size_asset.size

        top_left_x = screen_x - hc_size[0] // 2
        top_left_y = screen_y - hc_size[1] // 2

        base_ball_canvas = Image.new('RGBA', hc_size, (0, 0, 0, 0))
        tint_color = combo_colors[obj['combo_color_index']] if combo_colors else None
        is_digit_fallback = assets.get('is_digit_fallback', False)

        # If it's a digit fallback skin, the slider ball should be a simple tinted circle, not a number.
        if is_digit_fallback:
            if hitcircle_img:  # This should be the default skin's plain hitcircle
                tinted_ball_circle = tint_image(hitcircle_img, tint_color)
                base_ball_canvas = Image.alpha_composite(base_ball_canvas, tinted_ball_circle)
        # Otherwise, render the slider ball like a normal hitcircle (tinted base + overlay).
        else:
            if hitcircle_img:
                tinted_ball_circle = tint_image(hitcircle_img, tint_color)
                base_ball_canvas = Image.alpha_composite(base_ball_canvas, tinted_ball_circle)
            if overlay_img:
                base_ball_canvas = Image.alpha_composite(base_ball_canvas, overlay_img)

        frame_image.paste(base_ball_canvas, (top_left_x, top_left_y), base_ball_canvas)


def render_slider_reverse_arrow(frame_image, assets, obj, circle_pixel_radius, curve, at_start, raw_annotations):
    arrow_resized = assets.get('reversearrow')
    if not arrow_resized or obj['slides'] <= 1: return

    if at_start:
        arrow_pos, prev_pos = curve.nodes[:, 0], curve.nodes[:, 1]
    else:
        arrow_pos, prev_pos = curve.nodes[:, -1], curve.nodes[:, -2]
    angle = math.degrees(math.atan2(arrow_pos[1] - prev_pos[1], arrow_pos[0] - prev_pos[0])) + 180

    arrow_rotated = arrow_resized.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)
    screen_x = int(arrow_pos[0] * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X)
    screen_y = int(arrow_pos[1] * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)
    top_left_x = screen_x - arrow_rotated.width // 2
    top_left_y = screen_y - arrow_rotated.height // 2
    frame_image.paste(arrow_rotated, (top_left_x, top_left_y), arrow_rotated)


def render_slider(frame_image, assets, obj, difficulty, combo_colors, raw_annotations, slider_duration, is_hd,
                  is_tracked, main_font, use_scaling_animation=False):
    ar_ms = get_ar_ms(difficulty.get('ApproachRate', 9))
    current_time_ms = obj['current_time_ms']
    slider_end_time = obj['time'] + slider_duration

    head_opacity = 0.0
    body_opacity = 1.0

    if current_time_ms < obj['time']:
        fade_in_start_time = obj['time'] - ar_ms
        if current_time_ms >= fade_in_start_time:
            if is_hd:
                fade_in_duration = ar_ms * cfg.HD_FADE_IN_DURATION_MULTIPLIER
                fade_in_end_time = fade_in_start_time + fade_in_duration
                pre_hit_fade_out_duration = ar_ms - fade_in_duration
                if current_time_ms < fade_in_end_time:
                    raw_opacity = (current_time_ms - fade_in_start_time) / fade_in_duration
                    head_opacity = raw_opacity * 0.95 + 0.05
                else:
                    raw_opacity = (obj[
                                       'time'] - current_time_ms) / pre_hit_fade_out_duration if pre_hit_fade_out_duration > 0 else 0.0
                    head_opacity = raw_opacity * 0.95 + 0.05
            else:
                raw_opacity = (current_time_ms - fade_in_start_time) / ar_ms
                head_opacity = raw_opacity * 0.95 + 0.05

    if is_hd:
        time_since_hit = current_time_ms - obj['time']
        fade_in_duration = ar_ms * cfg.HD_FADE_IN_DURATION_MULTIPLIER
        fade_in_start_time = obj['time'] - ar_ms
        fade_in_end_time = fade_in_start_time + fade_in_duration
        if time_since_hit >= 0:
            post_hit_fade_duration = cfg.HD_SLIDER_BODY_FADE_OUT_DURATION_MS
            body_opacity = max(0.0, 1.0 - (time_since_hit / post_hit_fade_duration))
        else:
            if current_time_ms < fade_in_start_time:
                body_opacity = 0.0
            elif current_time_ms < fade_in_end_time:
                body_opacity = (current_time_ms - fade_in_start_time) / fade_in_duration
            else:
                body_opacity = 1.0
    else:
        fade_in_start_time = obj['time'] - ar_ms
        if current_time_ms < fade_in_start_time:
            body_opacity = 0.0
        elif current_time_ms < obj['time']:
            body_opacity = (current_time_ms - fade_in_start_time) / ar_ms
        else:
            body_opacity = 1.0
        time_after_slider_end = current_time_ms - slider_end_time
        if time_after_slider_end > 0:
            body_opacity = max(0.0, 1.0 - (time_after_slider_end / cfg.FADE_OUT_DURATION_MS))

    curve = obj.get('curve')
    if not curve: return

    playfield_scale = cfg.PLAYFIELD_HEIGHT / 384.0
    cs = difficulty.get('CircleSize', 4)
    circle_pixel_radius = int(get_cs_radius(cs) * playfield_scale)
    render_x, render_y = obj.get('render_x', obj['x']), obj.get('render_y', obj['y'])
    screen_x, screen_y = int(render_x * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X), int(
        render_y * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)

    if body_opacity > 0:
        body_canvas = Image.new('RGBA', frame_image.size, (0, 0, 0, 0))
        body_draw = ImageDraw.Draw(body_canvas)
        border_radius, fill_radius = circle_pixel_radius, int(circle_pixel_radius * 0.90)
        border_color, fill_color = (211, 211, 211, 200), (0, 0, 0, 70)
        path_points = [curve.evaluate(t) for t in np.linspace(0, 1, 75)]
        screen_path = [(int(p[0, 0] * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X),
                        int(p[1, 0] * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)) for p in path_points]

        for sx, sy in screen_path:
            body_draw.ellipse((sx - border_radius, sy - border_radius, sx + border_radius, sy + border_radius),
                              fill=border_color)
        for sx, sy in screen_path:
            body_draw.ellipse((sx - fill_radius, sy - fill_radius, sx + fill_radius, sy + fill_radius), fill=fill_color)

        render_slider_ticks(body_canvas, assets, obj, circle_pixel_radius, curve, body_opacity, raw_annotations)
        if body_opacity < 1.0: body_canvas.putalpha(body_canvas.getchannel('A').point(lambda p: int(p * body_opacity)))
        frame_image.paste(body_canvas, (0, 0), body_canvas)

    if head_opacity > 0:
        overlay_img = assets.get('hitcircleoverlay')
        hitcircle_img = assets.get('hitcircle')
        base_size_asset = overlay_img if overlay_img else hitcircle_img
        if not base_size_asset: return
        hc_size = base_size_asset.size

        base_head = Image.new('RGBA', hc_size, (0, 0, 0, 0))
        tint_color = combo_colors[obj['combo_color_index']] if combo_colors else None
        is_digit_fallback = assets.get('is_digit_fallback', False)

        if overlay_img:
            # Apply the same logic as above for the slider head
            img_to_paste = overlay_img if is_digit_fallback else tint_image(overlay_img, tint_color)
            if img_to_paste:
                base_head.paste(img_to_paste, (0, 0), img_to_paste)

        # Only render the hitcircle base if it's NOT a digit fallback.
        if hitcircle_img and not is_digit_fallback:
            tinted_head_circle = tint_image(hitcircle_img, tint_color)
            base_head = Image.alpha_composite(base_head, tinted_head_circle)

        render_combo_number(base_head, assets, obj['combo_number'], circle_pixel_radius, main_font)

        alpha = base_head.getchannel('A')
        base_head.putalpha(alpha.point(lambda p: int(p * head_opacity)))

        top_left_x_paste = screen_x - hc_size[0] // 2
        top_left_y_paste = screen_y - hc_size[1] // 2
        frame_image.paste(base_head, (top_left_x_paste, top_left_y_paste), base_head)

        # --- FIX: Use pre-calculated opaque dimensions for annotation ---
        opaque_dims = assets.get('hitcircle_opaque_dims')
        if opaque_dims and opaque_dims[0] is not None:
            box_w, box_h = opaque_dims
        else:
            box_w, box_h = hc_size

        top_left_x_anno = screen_x - box_w // 2
        top_left_y_anno = screen_y - box_h // 2
        raw_annotations.append({'class_id': cfg.CLASS_MAP['hit_circle'],
                                'box': [top_left_x_anno, top_left_y_anno, box_w, box_h]})
        # --- END FIX ---

        approachcircle_img = assets.get('approachcircle')
        if not is_hd and approachcircle_img:
            ac_opacity = head_opacity
            ac_scale = 1 + 3 * ((obj['time'] - current_time_ms) / ar_ms)
            ac_size = (int(hc_size[0] * ac_scale), int(hc_size[1] * ac_scale))
            if ac_size[0] > 0 and ac_size[1] > 0:
                ac_resized = approachcircle_img.resize(ac_size, Image.Resampling.LANCZOS)
                tinted_ac = tint_image(ac_resized, tint_color)
                if ac_opacity < 1.0: tinted_ac.putalpha(
                    tinted_ac.getchannel('A').point(lambda p: int(p * ac_opacity)))
                ac_top_left_x, ac_top_left_y = screen_x - ac_size[0] // 2, screen_y - ac_size[1] // 2
                frame_image.paste(tinted_ac, (ac_top_left_x, ac_top_left_y), tinted_ac)

    if 'hit_time' in obj and slider_duration > 0 and 0 <= current_time_ms - obj['time'] <= slider_duration:
        render_slider_ball(frame_image, assets, obj, combo_colors, slider_duration, circle_pixel_radius, curve,
                           raw_annotations, is_tracked, main_font)

        current_slide = math.floor(((current_time_ms - obj['time']) / slider_duration) * obj['slides'])
        if current_slide < obj['slides'] - 1:
            render_slider_reverse_arrow(frame_image, assets, obj, circle_pixel_radius, curve,
                                        not (current_slide % 2 == 0), raw_annotations)


def render_spinner(frame_image, assets, obj, difficulty, raw_annotations):
    spinner_bg, spinner_ac, spinner_circle = assets.get('spinner-background'), assets.get(
        'spinner-approachcircle'), assets.get('spinner-circle')
    center_x, center_y = cfg.OUTPUT_RESOLUTION[0] // 2, cfg.OUTPUT_RESOLUTION[1] // 2

    if spinner_bg: frame_image.paste(spinner_bg.resize(cfg.OUTPUT_RESOLUTION, Image.Resampling.LANCZOS),
                                     (center_x - spinner_bg.width // 2, center_y - spinner_bg.height // 2),
                                     spinner_bg.resize(cfg.OUTPUT_RESOLUTION, Image.Resampling.LANCZOS))
    if spinner_circle:
        target_h = int(cfg.PLAYFIELD_HEIGHT * 0.8)
        sc_w, sc_h = spinner_circle.size
        resized_circle = spinner_circle.resize((int(sc_w * (target_h / sc_h)), target_h),
                                               Image.Resampling.LANCZOS) if sc_h > 0 else spinner_circle
        rotated_circle = resized_circle.rotate(-(obj['current_time_ms'] * 0.4) % 360, expand=True,
                                               resample=Image.Resampling.BICUBIC)
        frame_image.paste(rotated_circle, (center_x - rotated_circle.width // 2, center_y - rotated_circle.height // 2),
                          rotated_circle)
    if spinner_ac:
        progress = max(0, min(1, (obj['current_time_ms'] - obj['time']) / (obj['end_time'] - obj['time']))) if (obj[
                                                                                                                    'end_time'] -
                                                                                                                obj[
                                                                                                                    'time']) > 0 else 1
        current_scale = 1.0 - progress
        ac_w, ac_h = spinner_ac.size
        if ac_h > 0 and current_scale > 0:
            initial_size = (int(ac_w * (cfg.PLAYFIELD_HEIGHT / ac_h)), cfg.PLAYFIELD_HEIGHT)
            current_size = (int(initial_size[0] * current_scale), int(initial_size[1] * current_scale))
            if current_size[0] > 0 and current_size[1] > 0:
                ac_resized = spinner_ac.resize(current_size, Image.Resampling.LANCZOS)
                frame_image.paste(ac_resized, (center_x - ac_resized.width // 2, center_y - ac_resized.height // 2),
                                  ac_resized)


def render_frame(frame_number, hit_objects, assets, difficulty, combo_colors, cursor_pos, is_hd,
                 game_simulation_state, render_ui, key_state, background_image=None, background_opacity=0.1,
                 main_font=None, use_scaling_animation=False, render_objects=True):
    global displayed_hp, cursor_trail_history

    if game_simulation_state is None:
        game_simulation_state = {}

    if background_image and background_opacity > 0:
        resized_bg = background_image.resize(cfg.OUTPUT_RESOLUTION, Image.Resampling.LANCZOS)
        frame_image = Image.blend(Image.new('RGBA', cfg.OUTPUT_RESOLUTION, (0, 0, 0, 255)), resized_bg,
                                  alpha=background_opacity)
    else:
        frame_image = Image.new('RGBA', cfg.OUTPUT_RESOLUTION, color=(0, 0, 0, 255))

    playfield_bg = Image.new('RGBA', (cfg.PLAYFIELD_WIDTH, cfg.PLAYFIELD_HEIGHT), color=(0, 0, 0, 80))
    frame_image.paste(playfield_bg, (cfg.PLAYFIELD_OFFSET_X, cfg.PLAYFIELD_OFFSET_Y), playfield_bg)

    raw_annotations = []
    current_time_ms = frame_number * 1000 / cfg.FRAME_RATE

    if render_objects:
        ar_ms = get_ar_ms(difficulty.get('ApproachRate', 9))

        for i in range(1, len(hit_objects)):
            prev_obj = hit_objects[i - 1]
            current_obj = hit_objects[i]

            if prev_obj.get('is_spinner') or current_obj.get('new_combo'):
                continue

            render_follow_points(frame_image, assets, prev_obj, current_obj, current_time_ms, ar_ms)

        visible_objects = []
        for obj in hit_objects:
            is_spinner = obj.get('is_spinner', False)
            end_time = obj.get('end_time',
                               obj.get('time') + obj.get('slider_duration', 0)) if not is_spinner else obj.get(
                'end_time', obj['time'])

            if not (obj['time'] - ar_ms - 100 < current_time_ms < end_time + 500):
                continue

            obj['current_time_ms'] = current_time_ms
            visible_objects.append(obj)

        visible_objects.sort(key=lambda x: x['time'], reverse=True)

        for obj in visible_objects:
            is_slider, is_spinner = obj.get('is_slider', False), obj.get('is_spinner', False)
            if is_slider:
                render_slider(frame_image, assets, obj, difficulty, combo_colors, raw_annotations,
                              obj.get('slider_duration', 0), is_hd,
                              game_simulation_state.get('is_slider_tracked', False),
                              main_font, use_scaling_animation)
            elif is_spinner:
                render_spinner(frame_image, assets, obj, difficulty, raw_annotations)
            else:
                render_hittable_object(frame_image, assets, obj, difficulty, combo_colors, raw_annotations, is_hd,
                                       main_font, use_scaling_animation)

        for obj in visible_objects:
            if 'miss_time' in obj:
                total_miss_anim_duration = cfg.MISS_FADE_IN_DURATION_MS + cfg.MISS_FADE_OUT_DURATION_MS
                time_since_miss = current_time_ms - obj['miss_time']

                if 0 < time_since_miss < total_miss_anim_duration:
                    miss_asset = assets.get('hit0')
                    if miss_asset:
                        opacity = 0.0
                        if time_since_miss < cfg.MISS_FADE_IN_DURATION_MS:
                            opacity = time_since_miss / cfg.MISS_FADE_IN_DURATION_MS
                        else:
                            time_into_fade_out = time_since_miss - cfg.MISS_FADE_IN_DURATION_MS
                            opacity = 1.0 - (time_into_fade_out / cfg.MISS_FADE_OUT_DURATION_MS)

                        opacity = max(0.0, min(1.0, opacity))
                        if opacity <= 0: continue

                        physics_progress = time_since_miss / total_miss_anim_duration
                        angle = physics_progress * cfg.MISS_ROTATION_DEGREES
                        rotated_miss_asset = miss_asset.rotate(-angle, expand=True, resample=Image.Resampling.BICUBIC)

                        final_miss_asset = rotated_miss_asset.copy()
                        if final_miss_asset.mode == 'RGBA':
                            alpha = final_miss_asset.getchannel('A')
                            final_miss_asset.putalpha(alpha.point(lambda p: int(p * opacity)))

                        render_x, render_y = obj.get('render_x', obj['x']), obj.get('render_y', obj['y'])
                        screen_x, screen_y = int(
                            render_x * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X), int(
                            render_y * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y)

                        y_offset = physics_progress * cfg.MISS_GRAVITY_DISTANCE_PX

                        top_left_x_paste = screen_x - final_miss_asset.width // 2
                        top_left_y_paste = screen_y - final_miss_asset.height // 2 + int(y_offset)

                        frame_image.paste(final_miss_asset, (top_left_x_paste, top_left_y_paste), final_miss_asset)

                        # --- FIX: Use pre-calculated opaque dimensions for annotation ---
                        opaque_dims = assets.get('hit0_opaque_dims')
                        if opaque_dims and opaque_dims[0] is not None:
                            box_w, box_h = opaque_dims
                        else:
                            box_w, box_h = miss_asset.size  # Fallback to un-rotated base asset size

                        top_left_x_anno = screen_x - box_w // 2
                        top_left_y_anno = screen_y - box_h // 2 + int(y_offset)  # Apply gravity to anno box too
                        raw_annotations.append({'class_id': cfg.CLASS_MAP['hit_miss'],
                                                'box': [top_left_x_anno, top_left_y_anno, box_w, box_h]})
                        # --- END FIX ---

    if cursor_pos:
        cursor_screen_pos = (int(cursor_pos[0] * (cfg.PLAYFIELD_WIDTH / 512.0) + cfg.PLAYFIELD_OFFSET_X),
                             int(cursor_pos[1] * (cfg.PLAYFIELD_HEIGHT / 384.0) + cfg.PLAYFIELD_OFFSET_Y))

        cursor_trail_history.append(cursor_screen_pos)
        if len(cursor_trail_history) > CURSOR_TRAIL_LENGTH: cursor_trail_history.pop(0)

        scaled_cursor = assets.get('cursor')
        scaled_trail = assets.get('cursortrail')
        if scaled_cursor:
            if scaled_trail and len(cursor_trail_history) > 1:
                for i, pos in enumerate(reversed(cursor_trail_history[:-1])):
                    opacity = 1.0 - (i / len(cursor_trail_history))
                    temp_trail = scaled_trail.copy()
                    temp_trail.putalpha(temp_trail.getchannel('A').point(lambda p: int(p * opacity)))
                    frame_image.paste(temp_trail, (pos[0] - temp_trail.width // 2, pos[1] - temp_trail.height // 2),
                                      temp_trail)

            frame_image.paste(scaled_cursor, (cursor_screen_pos[0] - scaled_cursor.width // 2,
                                              cursor_screen_pos[1] - scaled_cursor.height // 2), scaled_cursor)

            opaque_dims = assets.get('cursor_opaque_dims')
            if opaque_dims and opaque_dims[0] is not None and opaque_dims[1] is not None:
                box_w, box_h = opaque_dims
            else:
                box_w, box_h = scaled_cursor.size

            top_left_x = cursor_screen_pos[0] - box_w // 2
            top_left_y = cursor_screen_pos[1] - box_h // 2
            raw_annotations.append({'class_id': cfg.CLASS_MAP['cursor'], 'box': [top_left_x, top_left_y, box_w, box_h]})

    if render_ui and game_simulation_state:
        target_hp = game_simulation_state.get('hp', 1.0)
        displayed_hp += (target_hp - displayed_hp) * 0.1
        hp_bar_img = ui_renderer.render_hp_bar(assets, displayed_hp)
        score_img, combo_img = ui_renderer.render_score_combo(assets, game_simulation_state.get('score', 0),
                                                              game_simulation_state.get('combo', 0))
        acc_img = ui_renderer.render_accuracy(assets, game_simulation_state.get('accuracy', 100.0))
        key_overlay_img = ui_renderer.render_key_overlay(assets, key_state, game_simulation_state)

        if hp_bar_img: frame_image.paste(hp_bar_img, (5, 5), hp_bar_img)
        if score_img:
            score_pos = (cfg.OUTPUT_RESOLUTION[0] - score_img.width - 25, 5)
            frame_image.paste(score_img, score_pos, score_img)
            if acc_img: frame_image.paste(acc_img,
                                          (cfg.OUTPUT_RESOLUTION[0] - acc_img.width - 25, 5 + score_img.height + 5),
                                          acc_img)
        if combo_img: frame_image.paste(combo_img, (20, cfg.OUTPUT_RESOLUTION[1] - combo_img.height - 20), combo_img)
        if key_overlay_img:
            key_pos_x = cfg.OUTPUT_RESOLUTION[0] - key_overlay_img.width - 20
            key_pos_y = (cfg.OUTPUT_RESOLUTION[1] - key_overlay_img.height) // 2
            frame_image.paste(key_overlay_img, (key_pos_x, key_pos_y), key_overlay_img)

    final_annotation_strings = []
    for ann in raw_annotations:
        box = ann['box']
        class_id = ann['class_id']
        x, y, w, h = box

        x_center_norm = (x + w / 2) / cfg.OUTPUT_RESOLUTION[0]
        y_center_norm = (y + h / 2) / cfg.OUTPUT_RESOLUTION[1]
        width_norm = w / cfg.OUTPUT_RESOLUTION[0]
        height_norm = h / cfg.OUTPUT_RESOLUTION[1]

        final_annotation_strings.append(f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}")

    return frame_image, "\n".join(final_annotation_strings)