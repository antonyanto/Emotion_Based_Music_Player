# Emotion_Based_Music_Player

### A Machine Learning based music player by detecting the user's current emotion and playing song.
#### Workflow of the project:
    -> Detect if any human face is there.
    -> If any, then identify the facial emotion.
    -> Find the best song for that facial emotion. (i.e If emotion is sad, then playing sad songs)
    -> Play the song and again identify the emotion
    -> if the emotion is changed, then change song also.
#### To run:
    1. Download some songs in the wave format (i.e .wav file)
    2. Now create folder "musics" in the project's directory and save those songs in this folder.
    3. Now download this machine learning model from this google drive's link https://drive.google.com/file/d/1GDtwC1RAm8x8FdfPod3K05sDobT5olgN/view?usp=sharing
    4. Now create folder under same directory as before as "pretrainde_models" and save this model in this folder.
    5. Open camera_web.py and update the song names in the function definition (i.e update song names in these functions happy_song, sad_song, neutral_song....)
    6. Install all the packages listed in requirements.txt
    7. Now you are all set and just run the app.py file.
    8. It will prompt you to open browser in the localhost.
    9. Show emotions in the webcam and enjoy your songs now.
