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
    with gr.Row(equal_height=True):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    '<a href="https://github.com/clement-pages/gryannote"><img src="https://repository-images.githubusercontent.com/744648524/e841cef0-fbd9-45b0-9bce-536a1822c7b1" alt="gryannote logo" width="220"/></a>',
                )
        with gr.Column(scale=6):
                gr.Markdown('<h1 style="font-size: 3em;">gryannote</h1>')  
                gr.Markdown("<h2>Make the audio labeling process easier and faster! </h2>")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "To use the component, start by loading or recording audio."
                "Then apply the diarization pipeline (here [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1))" 
                "or double-click directly on the waveform. The annotations produced can be edited."
                "You can also use keyboard shortcuts to speed things up! Finally, produced annotations can be saved by cliking on the downloading button in the RTTM component."
            )
            gr.Markdown()
            gr.Markdown()
            gr.Markdown('<img src="https://github.com/clement-pages/gryannote/blob/main/docs/assets/poster-interspeech.jpg?raw=true" alt="gryannote poster"/>')
        with gr.Column():
            audio_labeling = AudioLabeling(
                type="filepath",
                interactive=True,
            )

            gr.Markdown()
            gr.Markdown()

            run_btn = gr.Button("Run pipeline")

            rttm = RTTM()

            gr.Markdown(
                """| Shortcut                                      | Action                                                                |
                    | --------------------------------------------- | --------------------------------------------------------------------- |
                    | `SPACE`                                       | Toggle play / pause                                                   |
                    | `ENTER`                                       | Create annotation at current time                                     |
                    | `SHIFT + ENTER`                               | Split annotation at current time                                      |
                    | `A`, `B`, `C`, ..., `Z`                       | Set active label. If there is a selected annotation, update its label |
                    | `LEFT` or `RIGHT`                             | Edit start time of selected annotation (if any) or move time cursor   |
                    | `SHIFT + LEFT` or `SHIFT + RIGHT`             | Same, but faster                                                      |
                    |`ALT + LEFT` or `ALT + RIGHT`                  | Edit end time of selected annotation                                  |
                    | `SHIFT + ALT + LEFT` or `SHIFT + ALT + RIGHT` | Same, but faster                                                      |
                    | `TAB`                                         | Select next annotation                                                |
                    | `SHIFT + TAB`                                 | Select previous annotation                                            |
                    |`BACKSPACE`                                    | Delete selected annotation and select the previous one                |
                    |`DELETE` or `SHIFT + BACKSPACE`                | Delete selected region and select the next one                        |
                    |`ESC`                                          | Unselect selected annotation and / or label                           |
                    | `UP` or `DOWN`                                | Zoom in/out                                                           |
                    | `F2`                                          | Open settings for the active label                                    |
                    """
            )

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
