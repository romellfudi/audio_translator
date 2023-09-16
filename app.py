import sys
import os
# By using XTTS you agree to CPML license https://coqui.ai/cpml
os.environ["COQUI_TOS_AGREED"] = "1"

import gradio as gr
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
tts.to("cuda")

def predict(prompt, language, audio_file_pth, mic_file_path, use_mic):
    if use_mic == True:
        if mic_file_path is not None:
            speaker_wav=mic_file_path
        else:
            gr.Warning("Please record your voice with Microphone, or uncheck Use Microphone to use reference audios")
            return (
                None,
                None,
            ) 
            
    else:
        speaker_wav=audio_file_pth

    if len(prompt)<2:
        gr.Warning("Please give a longer prompt text")
        return (
                None,
                None,
            )
    try:   
        tts.tts_to_file(
            text=prompt,
            file_path="output.wav",
            speaker_wav=speaker_wav,
            language=language,
        )
    except RuntimeError as e :
        if "device-assert" in str(e):
            # cannot do anything on cuda device side error, need tor estart
            gr.Warning("Unhandled Exception encounter, please retry in a minute")
            print("Cuda device-assert Runtime encountered need restart")
            sys.exit("Exit due to cuda device-assert")
        else:
            raise e
        
    return (
        gr.make_waveform(
            audio="output.wav",
        ),
        "output.wav",
    )


title = "Romellfudi "

description = """
Romell D.Z.(@romellfudi) Software Engineer, Business Intelligence Analist and Data Scientist
<a href="https://github.com/romellfudi">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Github"></a>
</p>
"""

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="One or two sentences at a time is better",
            value="Hi there, I'm your new voice clone. Try your best to upload quality audio. Like my grandma used to say: 'Garbage in, garbage out!'",
        ),
        gr.Dropdown(
            label="Language",
            info="Select an output language for the synthesised speech",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cs",
                "ar",
                "zh-cn",
            ],
            max_choices=1,
            value="en",
        ),
        gr.Audio(
            label="Reference Audio",
            info="Click on the ✎ button to upload your own target speaker audio",
            type="filepath",
            value="examples/female.wav",
        ),
        gr.Audio(source="microphone",
                 type="filepath",
                 info="Use your microphone to record audio",
                 label="Use Microphone for Reference"),
        gr.Checkbox(label="Check to use Microphone as Reference", value=False),
    ],
    outputs=[
        gr.Video(label="Waveform Visual"),
        gr.Audio(label="Synthesised Audio"),
    ],
    title=title,
    description=description,
).queue().launch(debug=True, share = True)