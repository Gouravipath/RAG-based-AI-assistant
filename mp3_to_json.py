import whisper
import json
import os
audios = os.listdir("audios")
model = whisper.load_model("large-v2")
for audio in audios:
    number = audio.split("_")[0]
    title = audio.split("_")[1].split(".")[0]
    print(number, title)
    result = model.transcribe(audio= f"audios/{audio}", language = "hi", task = "translate", word_timestamps=False)

    chunks = []
    for segment in result["segments"]:
        chunks.append({"number": number,"title": title,"start": segment["start"],"End": segment["end"], "Text": segment["text"] })
    chunks_with_metadata = {"Chunks": chunks, "Text": result["text"]}

    with open(f"json/{audio}.json", "w") as f:
     json.dump(chunks_with_metadata,f)