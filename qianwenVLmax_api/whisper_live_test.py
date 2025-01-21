from whisper_live.client import TranscriptionClient
client = TranscriptionClient(
  "localhost",
  9090,
  is_multilingual=True,
  lang="hi",
  translate=True,
  model="small"
)
client()