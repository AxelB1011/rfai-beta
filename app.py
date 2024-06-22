from flask import Flask, request, jsonify, redirect, url_for, render_template, make_response
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
import time
from celery import Celery
import requests
import threading
# import shutil
import subprocess
import yt_dlp
import redis
from flask_cors import CORS

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'  # Folder to store uploaded videos
DETECT_SCRIPT = 'deepfake_detection_script.py'  # Script for deepfake detection

# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        channel_id = request.form.get("channel_id")
        if channel_id:
            app.config['CHANNEL_ID'] = channel_id
            # threading.Thread(target=check_new_videos, args=(channel_id,), daemon=True).start()
            return redirect(url_for("index"))
    return render_template("index.html")

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
CLIENT_SECRETS_FILE = "client_secret.json"
CHANNEL_ID = "UCbNnrnEgAZvdwYWUBkNumQg"  # Replace with the YouTube channel ID you want to monitor
#_mjzC5RJMEk video id

# Redis configuration
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

def get_authenticated_service():
    creds = None
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES, redirect_uri='http://localhost:5000/oauth2callback'
            )
            creds = flow.run_local_server(port=5000)
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build("youtube", "v3", credentials=creds)

youtube = get_authenticated_service()

# @app.route("/")
# def index():
#     return "YouTube Video Processing Service is running."

@app.route("/oauth2callback")
def oauth2callback():
    # Handle the OAuth callback
    return "OAuth authentication successful!"

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    video_id = data.get("video_id")
    if video_id:
        download_and_process_video.delay(video_id)

    return jsonify({"status": "received"})

def get_channel_name(channel_id):
    request = youtube.channels().list(
        part="snippet",
        id=channel_id
    )
    response = request.execute()
    channel_name = response["items"][0]["snippet"]["title"]
    return channel_name

# Global variable to control the monitoring loop
monitoring = False
thread = None

@app.route("/api/start_monitoring", methods=["POST"])
def start_monitoring():
    global monitoring, thread
    if not monitoring:
        monitoring = True
        thread = threading.Thread(target=check_new_videos)
        thread.start()
        response_data = {"message": "Monitoring started successfully"}
    else:
        response_data = {"message": "Monitoring is already running"}
    return jsonify(response_data)

@app.route("/api/stop_monitoring", methods=["POST"])
def stop_monitoring():
    global monitoring
    monitoring = False
    response_data = {"message": "Monitoring stopped successfully"}
    return jsonify(response_data)

latest_video_id = None

def check_new_videos():
    global latest_video_id
    while True:
        request = youtube.activities().list(part="snippet,contentDetails", channelId=CHANNEL_ID, maxResults=1)
        response = request.execute()
        if response["items"]:
            latest_activity = response["items"][0]
            if latest_activity["snippet"]["type"] == "upload":
                video_id = latest_activity["contentDetails"]["upload"]["videoId"]
                if video_id != latest_video_id:
                    latest_video_id = video_id
                    download_and_process_video.delay(True)
        cn = get_channel_name(CHANNEL_ID)
        print(f"Monitoring on for channel {cn}")
        # time.sleep(5)  # Check every 5 minutes
        for _ in range(30): 
            if not monitoring:
                print("Monitoring stopped")
                return
        # time.sleep(2)

@app.route("/api/video", methods=["POST"])
def api_video():
    data = request.json
    video_id = data.get("video_id")
    title = data.get("title")
    description = data.get("description")
    print(f"Received video details: ID={video_id}, Title={title}, Description={description}")
    # Here you can add code to process the received data further, e.g., store it in a database

    response = {
        "status": "success",
        "video_id": video_id
    }
    return make_response(jsonify(response), 200)
    # return jsonify({"status": "success", "video_id": video_id})

# def download_video(video_id):
#     # Get video details
#     request = youtube.videos().list(part="snippet,contentDetails", id=video_id)
#     response = request.execute()
#     video_details = response["items"][0]
#     title = video_details['snippet']['title']
    
#     print(f"Downloading video: {title}")
    
#     # Create output directory if it doesn't exist
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
#     # Define output path
#     output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}.mp4")
    
#     # Define yt-dlp options
#     ydl_opts = {
#         'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
#         'outtmpl': output_path
#     }
    
#     # Download video using yt-dlp
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         ydl.download([f'https://www.youtube.com/watch?v={video_id}'])


# def run_deepfake_detection(video_id):
#     video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{video_id}.mp4")
#     result = subprocess.run(["python", DETECT_SCRIPT, video_path], capture_output=True, text=True)
#     detection_result = result.stdout.strip()
#     # detection_result = detection_result[len(detection_result)-4:len(detection_result)]
#     print(f"Deepfake detection result for {video_id}: {detection_result}")
#     # Store detection result
#     with open(f'{video_id}_detection_result.txt', 'w') as f:
#         f.write(f'Detection Result: {detection_result}\n')
#     return detection_result

@app.route("/api/downandproc", methods=["POST"])
# Function to download video from YouTube and perform deepfake detection
def download_and_process_video(process=False):
    global latest_video_id
    data = request.json
    video_id = data.get("video_id")
    latest_video_id = video_id
    req = youtube.videos().list(part="snippet,contentDetails", id=video_id)
    response = req.execute()
    video_details = response["items"][0]
    title = video_details['snippet']['title']
    description = video_details['snippet']['description']
    print(f"Processing video: {title}")

    # # Store video details locally
    # with open(f'{video_id}.txt', 'w') as f:
    #     f.write(f'Title: {title}\n')
    #     f.write(f'Description: {description}\n')

    # Store video details in Redis
    redis_client.hmset(video_id, {"title": title, "description": description})

    if process:
        return jsonify({"status": "new upload processed"})
    
    # Download video from YouTube
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(f'https://www.youtube.com/watch?v={video_id}', download=False)
            # print("Dict keys: ", info_dict.keys())
            video_url = info_dict.get('original_url')
        except Exception as e:
            return jsonify({"error": f"Failed to download video: {e}"})

    # Run deepfake detection
    # print("URL: ", video_url)
    result = subprocess.run(["python", DETECT_SCRIPT, video_url], capture_output=True, text=True)
    detection_result = result.stdout.strip()

    return jsonify({"status": "processed", "result": detection_result})
    # return jsonify({"status": "processed", "detection_result": detection_result})

# def download_and_process_video():
#     data = request.json
#     video_id = data.get("video_id")
#     # Download video from YouTube
#     download_video(video_id)
#     # Run deepfake detection
#     res = run_deepfake_detection(video_id)
#     return jsonify({"status": "processed", "detection_result": res})

if __name__ == "__main__":
    app.run(port=5000)
