import os
import shutil

def find_osu_songs_folder():
    """Tries to find the default osu! songs directory."""
    # Common paths for Windows
    local_app_data = os.getenv('LOCALAPPDATA')
    if local_app_data:
        win_path = os.path.join(local_app_data, 'osu!', 'Songs')
        if os.path.isdir(win_path):
            return win_path

    # Common paths for macOS
    home = os.path.expanduser('~')
    mac_path = os.path.join(home, 'Library', 'Application Support', 'osu!', 'Songs')
    if os.path.isdir(mac_path):
        return mac_path

    return None

def extract_background_images():
    """
    Goes through the osu! maps directory and saves all beatmap background
    images combined in a separate folder.
    """
    songs_path_input = input("Enter the path to your osu! Songs folder (or press Enter to try to autodetect): ")
    if not songs_path_input:
        songs_path = find_osu_songs_folder()
        if not songs_path:
            print("Could not automatically find your osu! Songs folder.")
            print("Please run the script again and provide the path.")
            return
        print(f"Detected osu! Songs folder at: {songs_path}")
    else:
        songs_path = songs_path_input

    if not os.path.isdir(songs_path):
        print(f"The directory '{songs_path}' does not exist.")
        return

    destination_folder = input("Enter the path to the folder where you want to save the backgrounds: ")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created destination folder: '{destination_folder}'")

    for folder_name in os.listdir(songs_path):
        beatmap_folder = os.path.join(songs_path, folder_name)
        if os.path.isdir(beatmap_folder):
            for file_name in os.listdir(beatmap_folder):
                if file_name.endswith(".osu"):
                    osu_file_path = os.path.join(beatmap_folder, file_name)
                    try:
                        with open(osu_file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if line.startswith("0,0,"):
                                    try:
                                        background_filename = line.split('"')[1]
                                        background_path = os.path.join(beatmap_folder, background_filename)
                                        if os.path.exists(background_path):
                                            new_filename = f"{folder_name}_{background_filename}"
                                            destination_path = os.path.join(destination_folder, new_filename)
                                            shutil.copy2(background_path, destination_path)
                                            print(f"Copied '{background_filename}' to '{destination_path}'")
                                            break  # Move to the next beatmap folder
                                    except IndexError:
                                        # Line does not contain a valid background entry
                                        pass
                    except Exception as e:
                        print(f"Could not process '{osu_file_path}': {e}")
                    break  # Assume one .osu file is enough to find the background

if __name__ == "__main__":
    extract_background_images()