"""
PDF OCR 模块 —— 缓存 + 并行处理 + 图片降采样
==============================================
提供 OCR 文本缓存和并行 OCR 能力，供 build_index.py 和 ingest_pdf.py 共用。

特性:
  - OCR 结果缓存到 pdf_ocr_cache.json（含图片 MD5，自动检测页面变更）
  - 多进程并行 OCR（ProcessPoolExecutor），Windows spawn 兼容
  - 图片自动降采样（max_size 参数），大幅减少 OCR 内存和耗时
  - 原子写入缓存文件，防止中断损坏
  - 首次 ~1-2 分钟（2 进程 + 降采样），后续秒级命中缓存

用法:
    from pdf_ocr import ocr_pages_parallel, load_ocr_cache, save_ocr_cache
    results = ocr_pages_parallel("pdf_pages", max_workers=2, max_image_size=1600)
    for filename, text, page_num in results:
        print(f"{filename}: {len(text)} chars")
"""

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

CACHE_PATH = "pdf_ocr_cache.json"
PAGES_DIR = "pdf_pages"


# ================================================================
# Cache I/O
# ================================================================

def load_ocr_cache() -> dict:
    """加载 OCR 缓存。失败或文件不存在返回空 dict。"""
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_ocr_cache(cache: dict) -> None:
    """原子写入 OCR 缓存（先写 .tmp 再替换，防止中断损坏）。"""
    tmp_path = CACHE_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, CACHE_PATH)
    except OSError:
        pass


# ================================================================
# Image preprocessing (resize to save memory & speed up OCR)
# ================================================================

def _resize_image(filepath: str, max_size: int) -> str:
    """
    将图片降采样到 max_size 以内，返回临时文件路径。
    如果不需要降采样或 PIL 不可用，返回原始路径。

    中文 OCR 对分辨率不敏感，1600px 足够识别 12pt+ 字体，
    但能把 3000×4000 的图从 ~8MB 压到 ~500KB，OCR 快 3-5 倍。
    """
    if not HAS_PIL or max_size <= 0:
        return filepath

    try:
        img = Image.open(filepath)
        w, h = img.size
        max_dim = max(w, h)
        if max_dim <= max_size:
            return filepath  # 已足够小，跳过

        # 等比例缩放
        ratio = max_size / max_dim
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)

        # 写入临时文件（与原图同目录，避免跨盘 IO）
        tmp_path = filepath + f"._rs{max_size}.png"
        img.save(tmp_path, "PNG", optimize=True)
        return tmp_path
    except Exception:
        return filepath


def _cleanup_resized(filepath: str) -> None:
    """删除降采样生成的临时文件。"""
    if filepath.endswith("._rs") or "._rs" in filepath:
        try:
            os.remove(filepath)
        except OSError:
            pass


# ================================================================
# Image hashing (detect page changes)
# ================================================================

def _compute_image_hash(filepath: str) -> str:
    """计算图片文件前 64KB 的 MD5，用于检测页面内容变化。"""
    try:
        with open(filepath, "rb") as f:
            return hashlib.md5(f.read(65536)).hexdigest()
    except OSError:
        return ""


# ================================================================
# Serial OCR (fallback)
# ================================================================

def _create_cnocr_engine():
    """创建 CnOcr 引擎实例（主进程/回退用）。"""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["CNOCR_DOWNLOAD_SOURCE"] = "CN"

    import cnstd.consts as _cnstd_c
    _cnstd_c.DOWNLOAD_SOURCE = "CN"

    from cnocr import CnOcr
    return CnOcr()


def ocr_single_page(filepath: str, engine=None) -> str:
    """
    对单张图片执行 OCR，返回纯文本（行间 \\n 分隔）。

    Args:
        filepath: 图片路径
        engine: CnOcr 实例（可选，不传则临时创建）

    Returns:
        OCR 文本字符串
    """
    if engine is None:
        engine = _create_cnocr_engine()
    try:
        results = engine.ocr(filepath)
        return "\n".join(d["text"] for d in results)
    except Exception:
        return ""


# ================================================================
# Parallel OCR worker (module-level, picklable)
# ================================================================

def _worker_ocr_chunk(args: tuple) -> dict:
    """
    子进程 OCR worker：处理一组页面图片，返回 {filename: text}。

    在 Windows spawn 模式下，此函数被序列化到子进程执行。
    必须在此函数内部设置环境变量并导入 CnOcr（不可在模块顶层）。

    Args:
        args: (chunk, max_image_size) 元组
            chunk: [(filepath, filename), ...]
            max_image_size: 图片最大边长，0 表示不缩放
    """
    chunk, max_image_size = args

    import os as _os

    # 设置国内镜像源（必须在 cnocr/cnstd 导入之前）
    _os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    _os.environ["CNOCR_DOWNLOAD_SOURCE"] = "CN"
    _os.environ["CNSTD_DOWNLOAD_SOURCE"] = "CN"

    # 强制 cnstd 读取 CN 下载源
    import cnstd.consts as _cnstd_c
    _cnstd_c.DOWNLOAD_SOURCE = "CN"

    from cnocr import CnOcr

    # 图片降采样（如果可用）
    _resize = None
    if max_image_size > 0:
        try:
            from PIL import Image as _PILImage
            def _resize(fp):
                try:
                    img = _PILImage.open(fp)
                    w, h = img.size
                    if max(w, h) > max_image_size:
                        ratio = max_image_size / max(w, h)
                        img = img.resize((int(w * ratio), int(h * ratio)), _PILImage.LANCZOS)
                        tmp = fp + f"._rs{max_image_size}.png"
                        img.save(tmp, "PNG", optimize=True)
                        return tmp
                except Exception:
                    pass
                return fp
        except ImportError:
            pass

    engine = CnOcr()
    results = {}
    for filepath, filename in chunk:
        ocr_path = filepath
        tmp_path = None
        try:
            if _resize:
                tmp_path = _resize(filepath)
                ocr_path = tmp_path if tmp_path != filepath else filepath
            out = engine.ocr(ocr_path)
            text = "\n".join(d["text"] for d in out)
            results[filename] = text
        except Exception:
            results[filename] = ""
        finally:
            # 清理降采样临时文件
            if tmp_path and tmp_path != filepath:
                try:
                    _os.remove(tmp_path)
                except OSError:
                    pass
    return results


# ================================================================
# Main public API
# ================================================================

def ocr_pages_parallel(
    page_dir: str = PAGES_DIR,
    max_workers: int = 2,
    use_cache: bool = True,
    force: bool = False,
    max_image_size: int = 1600,
) -> list:
    """
    并行 OCR 指定目录下的所有 PNG 页面，带缓存支持。

    Args:
        page_dir: 包含 page_*.png 的目录路径
        max_workers: 并行进程数（默认 2，Windows 上建议不超过 4）
        use_cache: 是否使用 OCR 缓存
        force: 强制重新 OCR 所有页面（忽略缓存）
        max_image_size: 图片最大边长（默认 1600px）。
                        设为 0 禁用降采样。
                        彩色图谱 1600px 足够识别中文，从 4000px 降到 1600px
                        OCR 快 ~4 倍，内存占用降 ~6 倍。

    Returns:
        [(filename, text, page_number), ...] 按页码排序
    """
    if not os.path.isdir(page_dir):
        return []

    # 1. 列出所有 PNG 页面
    page_files = sorted(
        [f for f in os.listdir(page_dir) if f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if not page_files:
        return []

    print(f"  找到 {len(page_files)} 页 PDF 图片")

    # 2. 加载缓存
    cache = load_ocr_cache() if use_cache and not force else {}
    cache_hits = 0
    uncached = []  # [(filepath, filename), ...]

    for fname in page_files:
        fpath = os.path.join(page_dir, fname)
        if fname in cache:
            cached_entry = cache[fname]
            cached_hash = cached_entry.get("image_hash", "")
            current_hash = _compute_image_hash(fpath)
            if cached_hash == current_hash and cached_entry.get("text", "").strip():
                cache_hits += 1
                continue
        uncached.append((fpath, fname))

    # 3. 并行 OCR 未缓存页面
    if uncached:
        total = len(uncached)
        print(f"  缓存命中 {cache_hits}/{len(page_files)}，需 OCR {total} 页"
              f"（{max_workers} 进程并行）...")

        t0 = time.time()
        try:
            _run_parallel_ocr(uncached, cache, page_dir, max_workers, max_image_size)
        except Exception as e:
            print(f"  并行 OCR 失败 ({e})，回退到串行模式...")
            _run_serial_ocr(uncached, cache, page_dir, max_image_size)

        elapsed = time.time() - t0
        rate = total / elapsed if elapsed > 0 else 0
        print(f"  OCR 完成：{total} 页 / {elapsed:.0f}s ({rate:.1f} 页/分)")

        # 保存缓存
        save_ocr_cache(cache)
    else:
        print(f"  缓存全部命中 ({cache_hits}/{len(page_files)} 页)，跳过 OCR")

    # 4. 组装返回结果
    results = []
    for fname in page_files:
        fpath = os.path.join(page_dir, fname)
        page_num = int(fname.split("_")[1].split(".")[0])

        if fname in cache:
            text = cache[fname].get("text", "")
        else:
            # 缓存未命中且未 OCR（不应该出现），回退串行
            print(f"  警告：{fname} 无缓存，正在回退 OCR...")
            text = ocr_single_page(fpath)
            cache[fname] = {
                "text": text,
                "image_hash": _compute_image_hash(fpath),
                "char_count": len(text),
                "ocr_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }

        results.append((fname, text, page_num))

    return results


def _run_parallel_ocr(
    uncached: list,
    cache: dict,
    page_dir: str,
    max_workers: int,
    max_image_size: int,
) -> None:
    """使用 ProcessPoolExecutor 并行 OCR。"""
    # Round-robin 分片，让各 worker 负载均衡
    chunks = [[] for _ in range(max_workers)]
    for i, item in enumerate(uncached):
        chunks[i % max_workers].append(item)

    # 去掉空 chunk
    chunks = [c for c in chunks if c]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker_ocr_chunk, (chunk, max_image_size)): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            try:
                worker_results = future.result()
                for filename, text in worker_results.items():
                    fpath = os.path.join(page_dir, filename)
                    cache[filename] = {
                        "text": text,
                        "image_hash": _compute_image_hash(fpath),
                        "char_count": len(text),
                        "ocr_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
            except Exception as e:
                print(f"  警告：一个 OCR worker 失败 ({e})，对应页面将回退串行处理")


def _run_serial_ocr(
    uncached: list,
    cache: dict,
    page_dir: str,
    max_image_size: int,
) -> None:
    """串行 OCR 回退（当并行失败时）。"""
    engine = _create_cnocr_engine()
    for i, (fpath, fname) in enumerate(uncached):
        # 降采样
        ocr_path = fpath
        tmp_path = None
        if max_image_size > 0:
            tmp_path = _resize_image(fpath, max_image_size)
            if tmp_path != fpath:
                ocr_path = tmp_path

        try:
            text = ocr_single_page(ocr_path, engine)
        finally:
            if tmp_path and tmp_path != fpath:
                _cleanup_resized(tmp_path)

        cache[fname] = {
            "text": text,
            "image_hash": _compute_image_hash(fpath),
            "char_count": len(text),
            "ocr_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if (i + 1) % 20 == 0:
            print(f"  OCR (串行)... {i+1}/{len(uncached)}")
