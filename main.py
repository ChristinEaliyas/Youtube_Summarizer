import gradio as gr
import yt_summarizer
import en_indic
import warnings

warnings.filterwarnings("ignore")
def display_video(link):
    return f'<iframe width="100%" height="500" src="https://www.youtube.com/embed/{link.split("?v=")[-1]}?autoplay=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>'


def pipeline(link, language):
    audio_path = yt_summarizer.download_youtube_audio(link)
    if audio_path:
        wav_path = yt_summarizer.convert_to_wav(audio_path)
        video_html = display_video(link)
        yield gr.update(value = video_html), gr.update(), gr.update() 
        transcription_output = yt_summarizer.transcribe(wav_path)
        yield gr.update(), gr.update(value = transcription_output), gr.update()
        summary_output = yt_summarizer.generate_summary(transcription_output)
        if language != "English":
            summary_output = en_indic.translate_sentence(summary_output, language)
            yield gr.update(), gr.update(), gr.update(value = summary_output)
        else:
            yield gr.update(), gr.update(), gr.update(value = summary_output)

        

with gr.Blocks() as demo:
    with gr.Row():
        video_input = gr.Textbox(label="Enter YouTube link")
        language = gr.Dropdown(choices=["English", "Hindi", "Malayalam", "Tamil", "Telugu"], label="Select Language")
        play_button = gr.Button("Play Video")

    with gr.Row():
        video_output = gr.HTML(label="Video will be displayed here")
        
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Transcription"):
                    transcription_output = gr.Textbox(
                        label="Transcription", interactive=False, elem_id="transcription_output"
                    )
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Summary"):
                    summary_output = gr.Textbox(
                        label="Summary", interactive=False, elem_id="summary_output"
                    )

    # Call the pipeline function when the button is clicked
    play_button.click(pipeline, inputs=[video_input, language], outputs=[video_output, transcription_output, summary_output])

demo.launch()
