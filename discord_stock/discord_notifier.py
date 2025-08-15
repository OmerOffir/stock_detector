import json
import requests
import time
from pathlib import Path

class DiscordNotifier:
    def __init__(self, config_path: str = "discord_stock\\webhook_meneger.json"):
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
        url = self.webhooks.get(channel_name)
        if not url:
            raise ValueError(f"No webhook URL found for channel '{channel_name}'")
        
        while True:
            r = requests.post(url, json={"embeds": [embed]}, timeout=20)
            if r.status_code in (200, 204):
                break
            if r.status_code == 429:
                data = r.json()
                retry = float(data.get("retry_after", 1))
                time.sleep(retry)
                continue
            raise Exception(f"Failed to send embed: {r.status_code} {r.text}")