# endpoints
GAMES_URL = "https://api.igdb.com/v4/games"


#error handling

MAX_RETRIES = 3  # Maximum number of retries


# auth config
# Real values live in IGDB_miner/.env (local only, excluded via .git/info/exclude).
from pathlib import Path

_env = {}
_env_file = Path(__file__).with_name(".env")
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _key, _value = _line.split("=", 1)
            _env[_key.strip()] = _value.strip()

token = _env.get("IGDB_TOKEN", "token here")
CLIENT_ID = _env.get("IGDB_CLIENT_ID", "client id here")
