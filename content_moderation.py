"""
content_moderation.py —— 百度内容审核（输入/输出安全校验）
==========================================================
百度智能云 text_censor/v2 接口：AK/SK 换 access_token → 文本审核。

- 结论映射：合规(1)→放行；不合规(2)→拦截；疑似(3)→放行+WARNING；审核失败(4)/超时→放行(fail-open)+ERROR
- access_token 内存缓存（提前 5 天过期即刷新），threading.Lock
- AK/SK 未配置 → NullCensor（直通 + 日志标注），本地调试不受阻

任务名（百度侧分任务看报表）：RAG_QA_INPUT（用户输入）/ RAG_QA_OUTPUT（模型输出）

用法:
    from content_moderation import build_censor
    censor = build_censor()
    result = censor.check_text("这是一段待审核文本", task="RAG_QA_INPUT")
    if not result.passed: ...  # 拦截
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import httpx

from config import BAIDU_AK, BAIDU_SK, BAIDU_CENSOR_ENABLED, CENSOR_TIMEOUT

_log = logging.getLogger("content_moderation")

_TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
_CENSOR_URL = "https://aip.baidubce.com/rest/2.0/solution/v1/text_censor/v2/user_defined"
_MAX_TEXT_BYTES = 20000  # 百度 text_censor 单次文本上限（字节）


@dataclass
class CensorResult:
    passed: bool
    conclusion: str = "合规"
    blocked_types: list = field(default_factory=list)
    raw: dict | None = None


class BaiduCensor:
    """百度文本审核。任何异常 fail-open（放行 + ERROR 日志，不阻断主链路）。

    免费档 QPS=1：流水线每请求 2 次审核（输入+输出），撞限流（error_code=18）时
    退避重试一次（默认 1.2s），重试耗尽仍限流 → fail-open 放行。
    """

    def __init__(self, api_key: str = "", secret_key: str = "", timeout: float = 3.0,
                 max_retries: int = 1, retry_backoff: float = 1.2):
        self._ak = api_key
        self._sk = secret_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._token = ""
        self._token_expire = 0.0
        self._lock = threading.Lock()

    def _get_access_token(self) -> str:
        with self._lock:
            if self._token and time.time() < self._token_expire:
                return self._token
            try:
                r = httpx.post(_TOKEN_URL, params={
                    "grant_type": "client_credentials",
                    "client_id": self._ak, "client_secret": self._sk,
                }, timeout=self._timeout)
                data = r.json()
            except Exception as e:
                _log.error("baidu_token_request_error err=%s", str(e)[:200])
                return ""
            token = data.get("access_token", "")
            if not token:
                _log.error("baidu_token_error resp=%s", str(data)[:200])
                return ""
            expires = int(data.get("expires_in", 2592000))
            self._token = token
            self._token_expire = time.time() + expires - 5 * 86400  # 提前 5 天刷新
            return token

    def check_text(self, text: str, task: str = "RAG_QA_INPUT") -> CensorResult:
        if not text or not text.strip():
            return CensorResult(passed=True)
        for attempt in range(self._max_retries + 1):
            try:
                token = self._get_access_token()
                if not token:
                    return CensorResult(passed=True)  # fail-open
                r = httpx.post(
                    _CENSOR_URL,
                    params={"access_token": token},
                    data={"text": text.encode("utf-8")[:_MAX_TEXT_BYTES].decode("utf-8", "ignore"),
                          "task_name": task},
                    timeout=self._timeout,
                )
                data = r.json()
            except Exception as e:
                _log.error("censor_exception task=%s err=%s", task, str(e)[:200])
                return CensorResult(passed=True)  # fail-open

            if "error_code" in data:
                # 免费档 QPS=1：限流时退避重试，重试耗尽仍失败 → fail-open
                if data.get("error_code") == 18 and attempt < self._max_retries:
                    _log.warning("censor_qps_retry task=%s attempt=%d", task, attempt + 1)
                    time.sleep(self._retry_backoff)
                    continue
                _log.error("baidu_censor_error task=%s code=%s msg=%s",
                           task, data.get("error_code"), data.get("error_msg"))
                return CensorResult(passed=True)  # fail-open
            break  # 拿到正常响应，跳出重试循环

        conclusion = data.get("conclusion", "审核失败")
        conclusion_type = data.get("conclusionType", 4)

        if conclusion_type == 1:
            return CensorResult(passed=True, conclusion="合规", raw=data)
        if conclusion_type == 2:
            types = [d.get("type") for d in data.get("data", [])]
            _log.warning("censor_blocked task=%s types=%s", task, types)
            return CensorResult(passed=False, conclusion="不合规", blocked_types=types, raw=data)
        # 3=疑似 → 放行 + 告警；4=审核失败 → fail-open
        _log.warning("censor_uncertain task=%s conclusion=%s type=%s",
                     task, conclusion, conclusion_type)
        return CensorResult(passed=True, conclusion=conclusion, raw=data)


class NullCensor(BaiduCensor):
    """AK/SK 未配置时的空实现：直通放行 + 日志标注。"""

    def __init__(self):
        super().__init__("", "")

    def check_text(self, text: str, task: str = "RAG_QA_INPUT") -> CensorResult:
        _log.info("censor_disabled task=%s", task)
        return CensorResult(passed=True, conclusion="未启用")


def build_censor():
    """按配置构建审核器：未启用或未配 AK/SK → NullCensor。"""
    if BAIDU_CENSOR_ENABLED and BAIDU_AK and BAIDU_SK:
        return BaiduCensor(BAIDU_AK, BAIDU_SK, CENSOR_TIMEOUT)
    return NullCensor()
