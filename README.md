Transkriberings App (faster-whisper, Tkinter)

## Hur man använder

1. Ladda ner senaste `.7z`-filen från Releases.
2. Packa upp den - högerklicka och välj "Extrahera alla", eller använd 7-Zip.
3. Öppna mappen `transcribe-app` och dubbelklicka på `transcribe-app.exe`.
4. Klicka på **Spela in** för att starta inspelningen av ett möte.
5. Klicka på **Stoppa** när mötet är klart - transkriberingen startar automatiskt.
6. När transkriberingen är klar visas texten i appen och sparas som en `.txt`-fil i mappen `transcripts` bredvid exe-filen.

Du kan också transkribera en befintlig ljudfil - klicka på **Välj ljudfil för transkribering...** och välj filen så startar transkriberingen direkt.

> Använder du `-nomodel`-versionen behöver du klicka på **Ladda ner modell** vid första start. Det kräver internetanslutning och tar några minuter. Efter det körs appen helt lokalt - ingen information skickas till internet.