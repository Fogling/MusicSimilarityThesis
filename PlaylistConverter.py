import spotipy
from spotipy.oauth2 import SpotifyOAuth
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import json
import time

# ---- SET THESE ----
SPOTIFY_CLIENT_ID = "your_spotify_client_id"
SPOTIFY_CLIENT_SECRET = "your_spotify_client_secret"
SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"
SPOTIFY_PLAYLIST_URL = "your_spotify_playlist_url"
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
    flow = InstalledAppFlow.from_client_secrets_file(YOUTUBE_CLIENT_SECRET_FILE, scopes)
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

def add_video_to_playlist(youtube, playlist_id, video_id):
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
    request.execute()

def main():
    sp = spotify_login()
    youtube = youtube_login()

    print("Fetching Spotify tracks...")
    tracks = get_spotify_tracks(sp, SPOTIFY_PLAYLIST_URL)
    print(f"Found {len(tracks)} tracks.")

    playlist_name = "Converted from Spotify"
    print(f"Creating YouTube playlist: {playlist_name}")
    yt_playlist_id = create_youtube_playlist(youtube, playlist_name)

    for query in tracks:
        print(f"Searching: {query}")
        video_id = search_youtube_video(youtube, query)
        if video_id:
            print(f"Adding video: https://youtube.com/watch?v={video_id}")
            add_video_to_playlist(youtube, yt_playlist_id, video_id)
        else:
            print("No match found.")
        time.sleep(1)  # to avoid quota issues

    print("Done!")

if __name__ == "__main__":
    main()
