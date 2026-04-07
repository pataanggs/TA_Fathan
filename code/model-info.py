import torchinfo as ti
import whisper

# load whisper model
whisper = whisper.load_model("base")
model = ti.summary(whisper)
print(model)
