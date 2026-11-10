"""路径与配置：全部基于 __file__ 解析为绝对路径，从任意工作目录启动均可正常工作。"""
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent

SEED_FILE = REPO_ROOT / "data" / "投资组合示例数据.json"
DIST_DIR = REPO_ROOT / "frontend" / "dist"
DB_PATH = BACKEND_DIR / "portfolio.db"
