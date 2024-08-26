import gradio as gr
from gryannote_audio import AudioLabeling
from gryannote_rttm import RTTM
from pyannote.audio import Pipeline
import os

def apply_pipeline(audio):
    """Apply specified pipeline on the indicated audio file"""
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=os.environ["HF_TOKEN"])
    annotations = pipeline(audio)

    return ((audio, annotations), (audio, annotations))


def update_annotations(data):
    return rttm.on_edit(data)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        '<a href="https://github.com/clement-pages/gryannote"><img src="https://github.com/clement-pages/gryannote/blob/main/docs/assets/logo-gryannote.png?raw=true" alt="gryannote logo" width="140"/></a>',
                        )
                with gr.Column(scale=10):
                        gr.Markdown('<h1 style="font-size: 4em;">gryannote</h1>')
                        gr.Markdown() 
                        gr.Markdown('<h2 style="font-size: 2em;">Make the audio labeling process easier and faster! </h2>')

            with gr.Tab("application"):
                gr.Markdown(
                    "To use the component, start by loading or recording audio."
                    "Then apply the diarization pipeline (here [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1))" 
                    "or double-click directly on the waveform to add an annotations. The annotations produced can be edited."
                    " You can also use keyboard shortcuts to speed things up! Click on the help button to see all the available shortcuts."
                    " Finally, annotations can be saved by cliking on the downloading button in the RTTM component."
                )
                gr.Markdown()
                gr.Markdown()
                audio_labeling = AudioLabeling(
                    type="filepath",
                    interactive=True,
                )

                gr.Markdown()
                gr.Markdown()

                run_btn = gr.Button("Run pipeline")

                rttm = RTTM()

            with gr.Tab("poster"):
                gr.Markdown('<img src="https://github.com/clement-pages/gryannote/blob/main/docs/assets/poster-interspeech.jpg?raw=true" alt="gryannote poster"/>')

    run_btn.click(
        fn=apply_pipeline,
        inputs=audio_labeling,
        outputs=[audio_labeling, rttm],
    )

    audio_labeling.edit(
        fn=update_annotations,
        inputs=audio_labeling,
        outputs=rttm,
        preprocess=False,
        postprocess=False,
    )


if __name__ == "__main__":
    demo.launch()
