A Python program to convert .wav Speech 2 text using OPENROUTER Whisper AI model.

Also added a Text to speech convertor useing a paid AI model called deepgram/aura-2 for $30/M characters. Only for english or french text to speech. Other languages are available but not listed. Check the deepgram docs for details.

First need an api key for the model Whisper AI at OPENROUTER.

Place it in a config.json file like this

{
    "OPENROUTER_API_KEY": "YOURAPI-KEY"
}
