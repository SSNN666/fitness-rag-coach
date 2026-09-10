"""
start.py —— 一键启动（API 后端 + Streamlit UI）
================================================
用法:
  python start.py                  # 启动全部 → 等就绪 → 自动打开浏览器
  python start.py --no-browser     # 不自动开浏览器
  python start.py --skip-api       # 仅 UI（知识图谱页签可用，问答需 API）
  python start.py --port-api 8000 --port-ui 8501

退出：Ctrl+C 同时关闭两个服务（子进程共享控制台信号）。
双击 start.bat 等价于在项目目录执行 python start.py。

设计要点:
  - Neo4j 图谱容器预检（NEO4J_ENABLED 时）：容器无 --restart，重启电脑后不会自启
    → 此处自动拉起；docker 不可用/容器不存在只提示不阻断（图谱有降级逻辑）
  - 启动顺序 API → UI（UI 是纯 SSE 客户端，先起后端避免冷启动空窗）
  - 健康检查等就绪（API healthz 最长 120s，pipeline/Milvus/适配器初始化需要时间）
  - 端口占用预检：已有实例在跑时给出提示而不是撞端口报错
  - 禁用 uvicorn --reload（Milvus Lite 单进程独占约束，见 README）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

# Windows GBK 控制台容错：print 含 ✓ 等 GBK 外字符时用 ? 替代而不是崩溃
# （双击 start.bat 场景实测踩坑：UnicodeEncodeError 'gbk' codec can't encode '✓'）
if sys.stdout.encoding and sys.stdout.encoding.lower() in ("gbk", "cp936"):
    sys.stdout.reconfigure(errors="replace")

ROOT = Path(__file__).parent


def _wait_http(url: str, timeout: float) -> bool:
    """轮询等 HTTP 200。超时返回 False。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _port_busy(port: int) -> bool:
    """端口是否已被监听（已有实例在跑时给友好提示）。"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


NEO4J_CONTAINER = "neo4j-fitness"


def _ensure_neo4j() -> None:
    """确保本地 Neo4j 容器在运行（仅 NEO4J_ENABLED 时）。

    为什么需要：容器创建时未设 --restart，**重启电脑后不会自动起来** →
    图谱页空白、三路检索静默退化成双路。演示前忘记拉起的代价很直接。

    失败不阻断启动：项目自身有图谱降级（禁忌数据走 contra_data 本地副本），
    docker 缺失/容器不存在都只提示不报错。
    """
    try:
        from config import NEO4J_ENABLED
    except Exception:
        return
    if not NEO4J_ENABLED:
        return

    def _docker(*argv: str):
        return subprocess.run(["docker", *argv], capture_output=True,
                              text=True, timeout=20)

    try:
        r = _docker("inspect", "-f", "{{.State.Running}}", NEO4J_CONTAINER)
        if r.returncode != 0:
            print(f"[!] 未找到 Neo4j 容器 {NEO4J_CONTAINER}，"
                  f"图谱将降级（禁忌数据自动走 contra_data 本地副本）")
            return
        if r.stdout.strip() == "true":
            print(f"[0/3] Neo4j 容器已在运行 ✓")
            return

        print(f"[0/3] 启动 Neo4j 容器 {NEO4J_CONTAINER} ...")
        _docker("start", NEO4J_CONTAINER)
        for _ in range(30):
            if _port_busy(7687):
                print("      Neo4j 就绪 ✓")
                return
            time.sleep(1)
        print("[!] Neo4j 启动后 30s 内未监听 7687，图谱可能不可用")
    except FileNotFoundError:
        print("[!] 未检测到 docker 命令，跳过 Neo4j 检查（图谱将降级）")
    except Exception as e:  # 超时/权限等一律不阻断主流程
        print(f"[!] Neo4j 检查异常（已忽略，不影响其他服务）: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="一键启动康养 RAG（API + Streamlit UI）")
    ap.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    ap.add_argument("--skip-api", action="store_true", help="仅启动 UI（图谱页签可用，问答需 API）")
    ap.add_argument("--port-api", type=int, default=8000)
    ap.add_argument("--port-ui", type=int, default=8501)
    args = ap.parse_args()

    # ---- 0. Neo4j 图谱容器（未运行则自动拉起，失败不阻断）----
    _ensure_neo4j()

    # 端口预检：已有健康实例时跳过本次启动（避免撞端口报错 / 重复拉起）
    if not args.skip_api and _port_busy(args.port_api):
        if _wait_http(f"http://127.0.0.1:{args.port_api}/healthz", 2):
            print(f"[!] 端口 {args.port_api} 已有健康 API 在运行，跳过本次 API 启动。")
            args.skip_api = True
            args.api_already_running = True
        else:
            print(f"[!] 端口 {args.port_api} 被非本项目的进程占用，API 启动可能失败。")
    if _port_busy(args.port_ui) and _wait_http(f"http://localhost:{args.port_ui}/_stcore/health", 2):
        print(f"[!] 端口 {args.port_ui} 已有 UI 在运行，跳过本次 UI 启动。")
        args.ui_running = True

    procs: list[subprocess.Popen] = []

    try:
        # ---- 1. API 后端 ----
        if not args.skip_api:
            print(f"[1/3] 启动 API 后端 (127.0.0.1:{args.port_api}) ...")
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "api:app",
                 "--host", "127.0.0.1", "--port", str(args.port_api)],
                cwd=ROOT,
            ))
            print("      等待 pipeline 就绪（Milvus + BM25 + 大模型适配器初始化）...")
            if _wait_http(f"http://127.0.0.1:{args.port_api}/healthz", 120):
                print("      API 就绪 ✓")
            else:
                print("[!] API 启动超时，请查看上方控制台输出排查")
        else:
            if getattr(args, "api_already_running", False):
                print("[1/3] API 已在运行，使用现有实例。")
            else:
                print("[1/3] 跳过 API（问答页签将不可用）")

        # ---- 2. Streamlit UI ----
        if getattr(args, "ui_running", False):
            print("[2/3] UI 已在运行，跳过启动。")
        else:
            print(f"[2/3] 启动 Streamlit UI (localhost:{args.port_ui}) ...")
            procs.append(subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "app.py",
                 "--server.headless", "true",
                 "--server.port", str(args.port_ui)],
                cwd=ROOT,
            ))
            _wait_http(f"http://localhost:{args.port_ui}/_stcore/health", 60)

        # ---- 3. 就绪 ----
        url = f"http://localhost:{args.port_ui}"
        print(f"[3/3] 全部就绪！UI: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        print("按 Ctrl+C 同时停止全部服务。")

        while True:
            # 任一子进程退出即视为结束（如 uvicorn 崩溃）
            for p in procs:
                if p.poll() is not None:
                    print("[!] 子进程异常退出，正在关闭其余服务...")
                    return
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n正在关闭服务 ...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        print("已退出，服务全部停止。")


if __name__ == "__main__":
    main()
