# Records assets/demo.gif, the before/after demo at the top of the README.
# Re-record from the repository root with:  vhs assets/demo.tape
# The models must already be downloaded (run lessons 8 and 9 once first).

Output assets/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 1280
Set Height 400
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 40ms

# Quiet output: models come from the local cache, no warnings or progress bars.
Env HF_HUB_OFFLINE "1"
Env TRANSFORMERS_VERBOSITY "error"
Env HF_HUB_DISABLE_PROGRESS_BARS "1"

Type 'uv run lessons/lesson09_rag.py "How do I run OpenSim locally?"'
Sleep 500ms
Enter
# Skip the ~10 s of model loading.
Hide
Wait+Screen@60s /Q: How do I run/
Show
Wait+Screen@60s /prompt tokens/
Sleep 4s

Hide
Type "clear"
Enter
Show

Type 'uv run lessons/lesson09_rag.py "Can I bring my car to Kestrel Island?"'
Sleep 500ms
Enter
Hide
Wait+Screen@60s /Q: Can I bring/
Show
Wait+Screen@60s /prompt tokens/
Sleep 5s
