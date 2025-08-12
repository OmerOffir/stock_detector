import json
import requests
from pathlib import Path

class DiscordNotifier:
    def __init__(self, config_path: str = "discord\\webhook_meneger.json"):
        self.webhooks = self._load_webhooks(config_path)

    def _load_webhooks(self, config_path: str) -> dict:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def send_message(self, channel_name: str, content: str):
        url = self.webhooks.get(channel_name)
        if not url:
            raise ValueError(f"No webhook URL found for channel '{channel_name}'")
        r = requests.post(url, json={"content": content}, timeout=20)
        if r.status_code != 204:
            raise Exception(f"Failed to send message: {r.status_code} {r.text}")

    def send_embed(self, channel_name: str, embed: dict):
        """Send a single embed (no image)."""
        url = self.webhooks.get(channel_name)
        if not url:
            raise ValueError(f"No webhook URL found for channel '{channel_name}'")
        r = requests.post(url, json={"embeds": [embed]}, timeout=20)
        if r.status_code not in (200, 204):
            raise Exception(f"Failed to send embed: {r.status_code} {r.text}")
