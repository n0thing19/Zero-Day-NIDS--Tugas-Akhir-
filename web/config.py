from pathlib import Path

# Jalur Direktori Utama
WEB_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_DIR.parent

# Jalur Artefak & Dataset
ARTIFACTS_DIR = ROOT_DIR / "notebooks" / "artifacts"
PROCESSED_DIR = ROOT_DIR / "dataset" / "processed"
BENIGN_TEST_DIR = PROCESSED_DIR / "benign_test_parts"
UNSEEN_ATTACK_DIR = PROCESSED_DIR / "unseen_attack_test_parts"

# Konfigurasi Default Aplikasi
DEFAULT_TAU = 0.9832
HOST = "127.0.0.1"
PORT = 8080