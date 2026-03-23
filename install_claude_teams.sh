#!/bin/bash

# Ensure the Claude settings directory exists
mkdir -p "$HOME/.claude"
SETTINGS_FILE="$HOME/.claude/settings.json"

if [ ! -f "$SETTINGS_FILE" ]; then
    echo '{"env": {"CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"}}' > "$SETTINGS_FILE"
    echo "Created ~/.claude/settings.json with agent teams enabled."
else
    # If the file already exists, it's safer to just export it in the bash profile
    # so we don't accidentally overwrite or malform existing JSON configuration.
    echo "export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1" >> "$HOME/.bashrc"
    echo "Appended CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS to ~/.bashrc."
    echo "Please run 'source ~/.bashrc' or restart your terminal for it to take effect."
fi
