import json
import os
import re
from typing import Dict, List, Any, Optional

class DialogueParser:
    def __init__(self, script_path: str):
        self.script_path = script_path
        self.data = self._load_script()

    def _load_script(self) -> Dict[str, Any]:
        """Load script from JSON or TXT file."""
        ext = os.path.splitext(self.script_path)[1].lower()
        
        if ext == '.json':
            return self._load_json()
        elif ext == '.txt':
            return self._load_txt()
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .json or .txt")

    def _load_json(self) -> Dict[str, Any]:
        """Load JSON format script."""
        with open(self.script_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle simple array format [{"text": "...", "speaker": "..."}]
        if isinstance(data, list):
            dialogues = data
            roles = set()
            
            # Normalize 'speaker' to 'role' and collect roles
            for dialogue in dialogues:
                if 'speaker' in dialogue and 'role' not in dialogue:
                    dialogue['role'] = dialogue.pop('speaker')
                if 'role' in dialogue:
                    roles.add(dialogue['role'])
            
            # Convert to full format
            data = {
                "metadata": {"title": "Imported from JSON", "default_language": "Chinese"},
                "speakers": {role: {} for role in roles},
                "dialogues": dialogues
            }
        # Handle full format with sections
        elif 'dialogues' in data:
            for dialogue in data['dialogues']:
                if 'speaker' in dialogue and 'role' not in dialogue:
                    dialogue['role'] = dialogue.pop('speaker')
        
        return data

    def _load_txt(self) -> Dict[str, Any]:
        """Load TXT format script (supports bracket and JSON-line formats)."""
        with open(self.script_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Detect format by checking first non-empty line
        first_line = next((l.strip() for l in lines if l.strip()), "")
        
        if first_line.startswith('['):
            return self._parse_bracket_format(lines)
        elif first_line.startswith('{'):
            return self._parse_json_line_format(lines)
        else:
            raise ValueError("Unknown TXT format. Expected '[role] text' or JSON-line format.")

    def _parse_bracket_format(self, lines: List[str]) -> Dict[str, Any]:
        """Parse [role] text format."""
        pattern = re.compile(r'^\[(.+?)\]\s*(.+)$')
        dialogues = []
        roles = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = pattern.match(line)
            if match:
                role, text = match.groups()
                roles.add(role)
                dialogues.append({"role": role, "text": text})
        
        return {
            "metadata": {"title": "Imported from TXT", "default_language": "Chinese"},
            "speakers": {role: {} for role in roles},  # Empty speaker configs
            "dialogues": dialogues
        }

    def _parse_json_line_format(self, lines: List[str]) -> Dict[str, Any]:
        """Parse JSON-line format (config line + text line)."""
        dialogues = []
        roles = set()
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Try to parse as JSON config
            if line.startswith('{'):
                try:
                    config = json.loads(line)
                    role = config.get("name", "Unknown")
                    
                    # Next line should be the dialogue text
                    i += 1
                    if i < len(lines):
                        text = lines[i].strip()
                        if text:
                            roles.add(role)
                            dialogues.append({
                                "role": role,
                                "text": text,
                                "config": config  # Store additional config
                            })
                except json.JSONDecodeError:
                    pass
            
            i += 1
        
        return {
            "metadata": {"title": "Imported from TXT", "default_language": "Chinese"},
            "speakers": {role: {} for role in roles},
            "dialogues": dialogues
        }

    def get_metadata(self) -> Dict[str, Any]:
        return self.data.get("metadata", {})

    def get_speakers(self) -> Dict[str, Any]:
        speakers = self.data.get("speakers", {})
        # Resolve paths relative to the script file if they are relative
        script_dir = os.path.dirname(os.path.abspath(self.script_path))
        for name, info in speakers.items():
            ref_audio = info.get("ref_audio")
            if ref_audio and not os.path.isabs(ref_audio):
                info["ref_audio"] = os.path.normpath(os.path.join(script_dir, ref_audio))
        return speakers

    def get_dialogues(self) -> List[Dict[str, Any]]:
        return self.data.get("dialogues", [])

    def get_roles(self) -> List[str]:
        """Get list of unique roles from dialogues."""
        return list(self.data.get("speakers", {}).keys())

    def validate(self, skip_audio_check: bool = False):
        """Validate script structure.
        
        Args:
            skip_audio_check: If True, skip checking if ref_audio files exist.
                             Useful for TXT format where audio paths are set later.
        """
        if "speakers" not in self.data:
            raise ValueError("Script must contain 'speakers' section.")
        if "dialogues" not in self.data:
            raise ValueError("Script must contain 'dialogues' section.")
        
        speakers = self.get_speakers()
        
        if not skip_audio_check:
            for name, info in speakers.items():
                if "ref_audio" not in info:
                    raise ValueError(f"Speaker '{name}' is missing 'ref_audio'.")
                if not os.path.exists(info["ref_audio"]):
                    raise ValueError(f"Reference audio for '{name}' does not exist: {info['ref_audio']}")

        dialogues = self.get_dialogues()
        for i, line in enumerate(dialogues):
            # Support both 'role' and 'speaker' keys
            role = line.get("role") or line.get("speaker")
            if not role:
                raise ValueError(f"Dialogue line {i} is missing 'role' or 'speaker'.")
            if role not in speakers:
                raise ValueError(f"Dialogue line {i} refers to unknown role '{role}'.")
            if "text" not in line:
                raise ValueError(f"Dialogue line {i} is missing 'text'.")

    def update_speaker_config(self, role: str, ref_audio: str, ref_text: str = "", language: str = "Chinese"):
        """Update speaker configuration (useful for TXT imports)."""
        if "speakers" not in self.data:
            self.data["speakers"] = {}
        
        self.data["speakers"][role] = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "language": language
        }
