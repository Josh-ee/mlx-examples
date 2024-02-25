import ffmpeg

def convert(input_file):
    # input_file = 'good_example.mp4'
    temp = input_file.split('.')[0]

    audio_file = f"{temp}.mp3"  # Define the path for the extracted audio

    # Load the video file
    input_file = ffmpeg.input(input_file)

    # Extract the audio and save it as an MP3 file
    input_file.output(audio_file, acodec='mp3').run()

# convert('videos/good_example.mp4')
    
