"""Step 0.1: check that every app ID in configs/apps.yaml exists on the Play Store."""
from pathlib import Path

import yaml
from google_play_scraper import app
from google_play_scraper.exceptions import NotFoundError

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "apps.yaml"


def main():
    with open(CONFIG, encoding="utf-8") as f:
        apps = yaml.safe_load(f)["apps"]

    ok = 0
    for entry in apps:
        try:
            info = app(entry["id"], lang="en", country="in")
            print(f"{entry['name']:<8} {entry['id']:<26} -> {info['title']}")
            ok += 1
        except NotFoundError:
            print(f"{entry['name']:<8} {entry['id']:<26} -> NOT FOUND (wrong ID?)")
        except Exception as e:
            print(f"{entry['name']:<8} {entry['id']:<26} -> ERROR: {e}")

    print(f"\n{ok}/{len(apps)} apps found")


if __name__ == "__main__":
    main()