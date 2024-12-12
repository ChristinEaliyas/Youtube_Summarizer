from flask import Flask, render_template, request
import yt_summarizer
import en_indic

app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('main.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    video_url = request.form['video_url']
    language = request.form['language']

    video_id = video_url.split('v=')[-1].split('&')[0]

    embed_url = f"https://www.youtube.com/embed/{video_id}"

    try:
        audio_path = yt_summarizer.download_youtube_audio(video_url)
        if not audio_path:
            raise Exception("Failed to download audio.")
        
        wav_path = yt_summarizer.convert_to_wav(audio_path)
        transcript = yt_summarizer.transcribe(wav_path)
        summary = yt_summarizer.generate_summary(transcript)

        if language != "English":
            summary = en_indic.translate_sentence(summary, language)

    except Exception as e:
        transcript = f"Error processing video: {str(e)}"
        summary = ""

    return render_template('result.html', 
                           embed_url=embed_url, 
                           transcript=transcript, 
                           summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
