import spotipy
from spotipy.oauth2 import SpotifyOAuth
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import json
import time
import sys

# ---- SET THESE ----
SPOTIFY_CLIENT_ID = "c8f4b187f2b745f1ba363a67f69b15b3"
SPOTIFY_CLIENT_SECRET = "8ae23b17695a4455b9ef9e603abfc06d"
SPOTIFY_REDIRECT_URI = "https://adb34eb28557.ngrok-free.app"
SPOTIFY_PLAYLIST_URL = "https://open.spotify.com/playlist/7MFwFHiJjC4unTZXQEzavy?si=84a6daccf9594525"
YOUTUBE_CLIENT_SECRET_FILE = "client_secrets.json"  # Downloaded from Google Cloud Console
# --------------------

def spotify_login():
    scope = "playlist-read-private"
    sp_oauth = SpotifyOAuth(client_id=SPOTIFY_CLIENT_ID,
                            client_secret=SPOTIFY_CLIENT_SECRET,
                            redirect_uri=SPOTIFY_REDIRECT_URI,
                            scope=scope)
    return spotipy.Spotify(auth_manager=sp_oauth)

def youtube_login():
    scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]
    flow = InstalledAppFlow.from_client_secrets_file(
        "client_secrets.json",
        scopes=scopes
    )
    credentials = flow.run_local_server(port=8080)
    return build("youtube", "v3", credentials=credentials)

def get_spotify_tracks(sp, playlist_url):
    playlist_id = playlist_url.split("/")[-1].split("?")[0]
    results = sp.playlist_tracks(playlist_id)
    tracks = []
    for item in results["items"]:
        track = item["track"]
        name = track["name"]
        artists = ", ".join([artist["name"] for artist in track["artists"]])
        search_query = f"{name} {artists}"
        tracks.append(search_query)
    return tracks

def create_youtube_playlist(youtube, title):
    request = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title": title,
                "description": "Auto-created from Spotify",
            },
            "status": {
                "privacyStatus": "private"
            }
        }
    )
    response = request.execute()
    return response["id"]

def get_existing_youtube_playlist_id(youtube, title):
    request = youtube.playlists().list(part="snippet", mine=True, maxResults=50)
    response = request.execute()
    for item in response.get("items", []):
        if item["snippet"]["title"] == title:
            return item["id"]
    return None

def search_youtube_video(youtube, query):
    request = youtube.search().list(
        part="snippet",
        q=query,
        type="video",
        maxResults=1
    )
    response = request.execute()
    items = response.get("items", [])
    if items:
        return items[0]["id"]["videoId"]
    return None

def add_video_to_playlist(youtube, playlist_id, video_id, retries=3):
    request = youtube.playlistItems().insert(
        part="snippet",
        body={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {
                    "kind": "youtube#video",
                    "videoId": video_id
                }
            }
        }
    )

    for attempt in range(1, retries + 1):
        try:
            request.execute()
            print(f"Successfully added video: https://youtube.com/watch?v={video_id}")
            break
        except Exception as e:
            print(f"[Attempt {attempt}] Error adding video {video_id}: {e}")
            if attempt == retries:
                print(f"Failed after {retries} attempts, skipping this video.")
                break
            time.sleep(2)

    time.sleep(5)

def main():
    sp = spotify_login()
    youtube = youtube_login()

    print("Fetching Spotify tracks...")
    tracks = get_spotify_tracks(sp, SPOTIFY_PLAYLIST_URL)
    print(f"Found {len(tracks)} tracks.")

    if len(sys.argv) > 1:
        playlist_name = sys.argv[1]
        yt_playlist_id = get_existing_youtube_playlist_id(youtube, playlist_name)
        if yt_playlist_id:
            print(f"Using existing playlist: {playlist_name}")
        else:
            print(f"No existing playlist found with name '{playlist_name}', creating a new one.")
            yt_playlist_id = create_youtube_playlist(youtube, playlist_name)
    else:
        playlist_name = "Zyzz Music"
        print(f"No playlist name provided. Creating new playlist: {playlist_name}")
        yt_playlist_id = create_youtube_playlist(youtube, playlist_name)

    start_index = int(input(f"Enter the start index (0-{len(tracks)-1:03}) or 0 to start from the beginning: "))
    max_tracks_per_run = 65
    processed_count = 0

    for idx, query in enumerate(tracks):
        if idx < start_index:
            continue

        if processed_count >= max_tracks_per_run:
            print(f"Reached daily limit of {max_tracks_per_run} tracks, stopping.")
            break

        print(f"[{idx}] Searching: {query}")
        video_id = search_youtube_video(youtube, query)
        if video_id:
            print(f"Adding video: https://youtube.com/watch?v={video_id}")
            add_video_to_playlist(youtube, yt_playlist_id, video_id)
        else:
            print("No match found.")

        processed_count += 1

    print("Done!")

if __name__ == "__main__":
    main()
