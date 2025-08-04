# skin_validator.py
#
# A utility to scan osu! skin folders, identify and delete unneeded files,
# and report missing or corrupted assets required by the dataset generator.
# cd source_files
# python skin_validator.py skins
# python skin_validator.py skins --delete

import os
import argparse
from PIL import Image
from tqdm import tqdm
import configparser

# --- Configuration ---
# This is the definitive list of base asset names that the dataset generator
# and renderer scripts might try to load. The script will keep both the
# standard and @2x versions of these files.
ESSENTIAL_ASSETS = {
    # Skin configuration
    "skin.ini",

    # Gameplay objects
    "hitcircle", "hitcircleoverlay", "approachcircle",
    "hit0", "hit50", "hit100", "hit300",
    "cursor", "cursortrail",
    "reversearrow",
    "sliderb",  # Slider body (fallback)
    "sliderfollowcircle",
    "sliderscorepoint", "slider-tick",

    # Spinners
    "spinner-approachcircle", "spinner-background", "spinner-circle",
    "spinner-glow", "spinner-middle", "spinner-top",

    # UI elements
    "scorebar-bg", "scorebar-colour",
    "inputoverlay-background", "inputoverlay-key", "inputoverlay-key-down",
}

# Prefixes for numbered assets that are dynamically loaded
NUMERIC_PREFIXES = ["default", "score", "combo"]


def get_full_asset_list():
    """Generates the full list of essential asset filenames."""
    full_list = set(ESSENTIAL_ASSETS)
    # Add all the numbered assets (e.g., score-0, combo-1, etc.)
    for prefix in NUMERIC_PREFIXES:
        for i in range(10):
            full_list.add(f"{prefix}-{i}")
    # Add assets that use prefixes but aren't numbers
    full_list.add("score-dot")
    full_list.add("score-percent")
    full_list.add("score-x")
    full_list.add("combo-x")
    return full_list


def validate_skin(skin_path, all_essential_assets):
    """
    Scans a single skin folder, classifies files, and checks for issues.

    Args:
        skin_path (str): The path to the skin folder.
        all_essential_assets (set): A set of all essential asset base names.

    Returns:
        A dictionary containing the validation results.
    """
    if not os.path.isdir(skin_path):
        return None

    files_to_keep = []
    files_to_delete = []
    unreadable_files = []
    ini_errors = []
    found_assets = set()

    # --- 1. Classify all existing files ---
    for filename in os.listdir(skin_path):
        file_path = os.path.join(skin_path, filename)
        if not os.path.isfile(file_path):
            continue

        base_name, ext = os.path.splitext(filename.lower())
        if base_name.endswith('@2x'):
            base_name = base_name[:-3]

        if base_name in all_essential_assets or filename.lower() == 'skin.ini':
            files_to_keep.append(file_path)
            found_assets.add(base_name)
        else:
            files_to_delete.append(file_path)

    # --- 2. Check for missing essential files ---
    if os.path.basename(skin_path).lower() != 'default':
        missing_files = all_essential_assets - found_assets
    else:
        missing_files = set()

    # --- 3. Validate skin.ini ---
    skin_ini_path = os.path.join(skin_path, 'skin.ini')
    if 'skin.ini' not in [os.path.basename(f).lower() for f in files_to_keep]:
        # Already handled by missing_files check, but good to be explicit
        pass
    else:
        try:
            # Use the same robust parsing method as skin_loader.py
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

            config = configparser.ConfigParser(strict=False)
            config.read_string(cleaned_content)
        except Exception as e:
            ini_errors.append(f"Failed to parse: {e}")

    # --- 4. Check for unreadable/corrupted image files ---
    for file_path in files_to_keep:
        if file_path.lower().endswith('.png'):
            try:
                with Image.open(file_path) as img:
                    img.load()
            except Exception as e:
                unreadable_files.append((os.path.basename(file_path), str(e)))

    return {
        "name": os.path.basename(skin_path),
        "path": skin_path,
        "keep": files_to_keep,
        "delete": files_to_delete,
        "missing": sorted(list(missing_files)),
        "unreadable": unreadable_files,
        "ini_errors": ini_errors
    }


def main():
    parser = argparse.ArgumentParser(
        description="A tool to clean and validate osu! skin folders for dataset generation.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("skins_dir", help="Path to the directory containing all skin folders.")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete unneeded files. Default is a dry run."
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip the final confirmation prompt before deleting files. Use with caution."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.skins_dir):
        print(f"[ERROR] Skins directory not found at: {args.skins_dir}")
        return

    all_essential_assets = get_full_asset_list()
    skin_folders = [os.path.join(args.skins_dir, d) for d in os.listdir(args.skins_dir) if
                    os.path.isdir(os.path.join(args.skins_dir, d))]

    print(f"Found {len(skin_folders)} skins to analyze. Starting validation...\n")

    all_results = []
    files_to_be_deleted_count = 0

    for skin_path in tqdm(skin_folders, desc="Analyzing Skins"):
        result = validate_skin(skin_path, all_essential_assets)
        if result:
            all_results.append(result)
            files_to_be_deleted_count += len(result['delete'])

    # --- Print all reports ---
    print("\n" + "=" * 50)
    print("SKIN VALIDATION REPORT")
    print("=" * 50 + "\n")

    for result in all_results:
        print(f"--- Skin: {result['name']} ---")
        has_issues = result['missing'] or result['unreadable'] or result['delete'] or result['ini_errors']

        if not has_issues:
            print("✅ Status: OK! All essential assets found and readable. No extra files.")
        else:
            if result['delete']:
                print(f"🗑️  Files to Delete: {len(result['delete'])}")

            if result['missing']:
                print(f"❓ Missing Assets: {len(result['missing'])}")
                for i, asset in enumerate(result['missing']):
                    if i < 5:
                        print(f"    - {asset}" + (".png" if asset != "skin.ini" else ""))
                if len(result['missing']) > 5:
                    print(f"    - ... and {len(result['missing']) - 5} more.")

            # --- NEW: Report skin.ini errors ---
            if result['ini_errors']:
                print(f"📄 skin.ini Issues: {len(result['ini_errors'])}")
                for error in result['ini_errors']:
                    print(f"    - {error}")

            if result['unreadable']:
                print(f"☠️  Unreadable/Corrupted Images: {len(result['unreadable'])}")
                for filename, error in result['unreadable']:
                    print(f"    - {filename}: {error}")
        print("-" * (len(result['name']) + 10) + "\n")

    if not args.delete:
        print("\n--- Dry Run Complete ---")
        print(f"A total of {files_to_be_deleted_count} files would be deleted.")
        print("To perform the deletion, run the script again with the --delete flag.")
        return

    if files_to_be_deleted_count == 0:
        print("\nNo files to delete. Your skins are clean!")
        return

    # --- Handle Actual Deletion ---
    print("\n" + "=" * 50)
    print("DELETION MODE ENABLED")
    print("=" * 50 + "\n")

    if not args.yes:
        confirm = input(
            f"You are about to permanently delete {files_to_be_deleted_count} files. This cannot be undone.\n"
            "Are you sure you want to continue? (y/n): "
        )
        if confirm.lower() != 'y':
            print("Deletion cancelled by user.")
            return

    print("Proceeding with file deletion...")
    total_deleted = 0
    with tqdm(total=files_to_be_deleted_count, desc="Deleting Files") as pbar:
        for result in all_results:
            for file_path in result['delete']:
                try:
                    os.remove(file_path)
                    total_deleted += 1
                except Exception as e:
                    print(f"\n[ERROR] Could not delete {file_path}: {e}")
                pbar.update(1)

    print(f"\n--- Deletion Complete ---")
    print(f"Successfully deleted {total_deleted} files.")


if __name__ == "__main__":
    main()