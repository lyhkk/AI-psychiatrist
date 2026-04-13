"""
webapp/app.py —— Flask 应用工厂与入口

模块分工
--------
  app.py              : Flask 创建、配置、蓝图注册、根路由
  routes/chat.py      : 对话 API 蓝图（/api/chat/*）
  core/session_manager: 会话管理，封装 LangGraph agents 系统
  templates/index.html: 单页前端
  static/             : CSS / JS 静态资源

解耦说明
--------
- 升级 agents 智能体系统：只改动 agents/ 包，webapp/core/session_manager.py
  仅需保证 build_intervention_graph / make_initial_state 接口不变。
- 修改 Flask 路由/前端：只改动 webapp/ 内文件，agents/ 完全不受影响。
- 修改评测管线：只改动 eval_pipeline.py，与 webapp 无任何耦合。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── 路径设置：确保项目根目录在 sys.path 首位 ──────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, render_template
from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")


# ─────────────────────────────────────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging(debug: bool = False) -> None:
    _LOG_DIR = _PROJECT_ROOT / "logs"
    _LOG_DIR.mkdir(exist_ok=True)

    from datetime import datetime
    log_file = _LOG_DIR / f"webapp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setFormatter(fmt)
    file_h.setLevel(logging.DEBUG)

    console_h = logging.StreamHandler()
    console_h.setFormatter(fmt)
    console_h.setLevel(logging.DEBUG if debug else logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_h)
    root.addHandler(console_h)


# ─────────────────────────────────────────────────────────────────────────────
# 应用工厂
# ─────────────────────────────────────────────────────────────────────────────

def create_app(debug: bool = False) -> Flask:
    """
    Flask 应用工厂。

    Parameters
    ----------
    debug : bool   是否开启调试模式（详细日志 + Flask reloader）
    """
    _setup_logging(debug)

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )

    # Session 密钥（生产环境请替换为随机长字符串或从 .env 读取）
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "cbt-discover-dev-secret-change-in-prod")
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

    # ── 注册蓝图 ──────────────────────────────────────────────────────────────
    from webapp.routes import chat_bp
    app.register_blueprint(chat_bp)

    # ── 根路由：返回单页前端 ──────────────────────────────────────────────────
    @app.get("/")
    def index():
        return render_template("index.html")

    logger = logging.getLogger("cbt.webapp")
    logger.info("[App] CBT-Discover Web Application initialized  debug=%s", debug)
    return app


# ─────────────────────────────────────────────────────────────────────────────
# 直接运行入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _debug = os.getenv("FLASK_DEBUG", "0") == "1"
    _port  = int(os.getenv("FLASK_PORT", "5000"))
    _host  = os.getenv("FLASK_HOST", "127.0.0.1")

    application = create_app(debug=_debug)
    print(f"""
╔══════════════════════════════════════════════════╗
║      CBT-Discover  心理辅助系统  Web 服务         ║
╠══════════════════════════════════════════════════╣
║  地址：http://{_host}:{_port:<5}                      ║
║  调试：{'开启' if _debug else '关闭'}                                ║
║  停止：Ctrl+C                                    ║
╚══════════════════════════════════════════════════╝
    """)
    application.run(host=_host, port=_port, debug=_debug, threaded=True)

