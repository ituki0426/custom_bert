# coding=utf-8
# Copyright 2020 Optuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Logging utilities."""

import functools
import logging
import os
import sys
import threading

# loggingによって出力されるログには、そのログの発生事由の"重大度"に応じて以下のようなレベルが設定される。
from logging import (
    CRITICAL,  # プログラム自体が実行を続けられないことを表す、重大なエラー。
    DEBUG,  # おもに問題を診断するときにのみ関心があるような、詳細な情報。
    ERROR,  # より重大な問題により、ソフトウェアがある機能を実行できないこと。
    FATAL,  # NOQA
    INFO,  # 想定された通りのことが起こったことの確認。
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,  # 想定外のことが起こった、または問題が近く起こりそうである (例えば、'disk space low') ことの表示。
)

from logging import captureWarnings as _captureWarnings
from typing import Optional

import huggingface_hub.utils as hf_hub_utils
from tqdm import auto as tqdm_lib

# ロックオブジェクトを作成する
# ロギング設定が複数のスレッドから同時にアクセスできないようにするためのオブジェクト。
_lock = threading.Lock()

# ハンドラ：ログの出力先を指定するためのオブジェクト
# StreamHandler：ログを標準出力やファイルに出力するためのハンドラ。
# FileHandler：ログをファイルに出力するためのハンドラである。
# RotatingFileHandler：ログをファイルに出力するためのハンドラ。ファイルサイズが一定値を超えると、ファイルをローテーションする。
# TimedRotatingFileHandler：ログをファイルに出力するためのハンドラ。一定時間ごとにファイルをローテーションする。
# NullHandler：ログを出力しないためのハンドラ。
# SMTPHandler：ログをメールで送信するためのハンドラ。
# HTTPHandler：ログをHTTPサーバに送信するためのハンドラ。
# SocketHandler：ログをソケットに送信するためのハンドラ。
# DatagramHandler：ログをUDPソケットに送信するためのハンドラ。
# SysLogHandler：ログをsyslogに送信するためのハンドラ。

_default_handler: Optional[logging.Handler] = None


log_levels = {
    "detail": logging.DEBUG,  # will also print filename and line number
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# logger.setLevel(logging.DEBUG)などのように、loggerのレベルを設定  
# 設定したレベル以上の重要度のログのみを出力
# この場合（WARNING, ERROR, CRITICAL）のみが出力されます。
_default_log_level = logging.WARNING

_tqdm_active = not hf_hub_utils.are_progress_bars_disabled()


def _get_default_logging_level():
    """
    環境変数 TRANSFORMERS_VERBOSITY の値を取得します。
    環境変数が設定されていない場合、デフォルト値 None を返します。
    Args:
        None
    Returns:
        int: ログレベル
    """
    # 環境変数が設定されていない場合、デフォルト値 None を返す
    env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
    if env_level_str:
        # 環境変数の値が log_levels に含まれている場合、その値を返す
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
        # 環境変数の値が log_levels に含まれていない場合、警告を出力
            logging.getLogger().warning(
                f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, "
                f"has to be one of: { ', '.join(log_levels.keys()) }"
            )
    return _default_log_level


def _get_library_name() -> str:
    """
    この関数が返す値
    1.ライブラリ名（トップレベルパッケージ名）：
        モジュール名が package.module.submodule形式のときpackageを返す
    2.スクリプトの場合：
        __main__を返す

    """
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    """
    1.ライブラリとして使用される場合：
        モジュール名が my_library.submodule.utilsのとき、_get_library_name() が返す値は "my_library" 
        よって、logging.getLogger("my_library")のようにしてロガーを取得する
    2.スクリプトとして使用される場合：
        __name__は "__main__" となるため、logging.getLogger("__main__")のようにしてロガーを取得する
    """
    # logging.getLogger(name):指定した名前（name）のロガーを取得します。
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler
    # 複数のスレッドがこの関数を同時に実行してロガー設定を変更することを防ぐ
    with _lock:
        # ライブラリのルートロガーがすでに設定されている場合
        if _default_handler:
            return
        # StreamHandler を使用してデフォルトのログ出力先を sys.stderr に設定
        _default_handler = logging.StreamHandler()
        # sys.stderr が None の場合、os.devnull を開いて出力先を設定
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")
        # ログの出力を即時反映させるために、sys.stderr の flush メソッドをハンドラーに関連付ける
        _default_handler.flush = sys.stderr.flush

        # ルートロガーの取得
        library_root_logger = _get_library_root_logger()
        # ハンドラーをルートロガーに追加
        library_root_logger.addHandler(_default_handler)
        # ルートロガーのレベルを設定
        library_root_logger.setLevel(_get_default_logging_level())
        # if logging level is debug, we add pathname and lineno to formatter for easy debugging
        if os.getenv("TRANSFORMERS_VERBOSITY", None) == "detail":
            formatter = logging.Formatter("[%(levelname)s|%(pathname)s:%(lineno)s] %(asctime)s >> %(message)s")
            _default_handler.setFormatter(formatter)
        # ログが親ロガーに伝播するのを防ぐ
        # これにより、ライブラリ固有のログ設定が外部の設定に影響を受けなくなる
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_log_levels_dict():
    return log_levels


def captureWarnings(capture):
    """
    Pythonの標準ライブラリである warnings モジュールが発する警告を logging モジュールを使ってログとして記録できるようにする
    """
    # py.warnings ロガー を使用して警告を記録します。
    logger = get_logger("py.warnings")

    if not logger.handlers:
        logger.addHandler(_default_handler)

    logger.setLevel(_get_library_root_logger().level)

    _captureWarnings(capture)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom transformers module.
    """

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """
    Return the current level for the 🤗 Transformers's root logger as an int.

    Returns:
        `int`: The logging level.

    <Tip>

    🤗 Transformers has following logging levels:

    - 50: `transformers.logging.CRITICAL` or `transformers.logging.FATAL`
    - 40: `transformers.logging.ERROR`
    - 30: `transformers.logging.WARNING` or `transformers.logging.WARN`
    - 20: `transformers.logging.INFO`
    - 10: `transformers.logging.DEBUG`

    </Tip>"""

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for the 🤗 Transformers's root logger.

    Args:
        verbosity (`int`):
            Logging level, e.g., one of:

            - `transformers.logging.CRITICAL` or `transformers.logging.FATAL`
            - `transformers.logging.ERROR`
            - `transformers.logging.WARNING` or `transformers.logging.WARN`
            - `transformers.logging.INFO`
            - `transformers.logging.DEBUG`
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """Set the verbosity to the `INFO` level."""
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Set the verbosity to the `WARNING` level."""
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """Set the verbosity to the `DEBUG` level."""
    return set_verbosity(DEBUG)


def set_verbosity_error():
    """Set the verbosity to the `ERROR` level."""
    return set_verbosity(ERROR)


def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def add_handler(handler: logging.Handler) -> None:
    """adds a handler to the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert handler is not None
    _get_library_root_logger().addHandler(handler)


def remove_handler(handler: logging.Handler) -> None:
    """removes given handler from the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert handler is not None and handler not in _get_library_root_logger().handlers
    _get_library_root_logger().removeHandler(handler)


def disable_propagation() -> None:
    """
    Disable propagation of the library log outputs. Note that log propagation is disabled by default.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """
    Enable propagation of the library log outputs. Please disable the HuggingFace Transformers's default handler to
    prevent double logging if the root logger has been configured.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True


def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every HuggingFace Transformers's logger. The explicit formatter is as follows:
    ```
        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)


def reset_format() -> None:
    """
    Resets the formatting for HuggingFace Transformers's loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        handler.setFormatter(None)


def warning_advice(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but if env var TRANSFORMERS_NO_ADVISORY_WARNINGS=1 is set, this
    warning will not be printed
    """
    no_advisory_warnings = os.getenv("TRANSFORMERS_NO_ADVISORY_WARNINGS", False)
    if no_advisory_warnings:
        return
    self.warning(*args, **kwargs)


logging.Logger.warning_advice = warning_advice


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once


@functools.lru_cache(None)
def info_once(self, *args, **kwargs):
    """
    This method is identical to `logger.info()`, but will emit the info with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.info(*args, **kwargs)


logging.Logger.info_once = info_once


class EmptyTqdm:
    """Dummy tqdm which doesn't do anything."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""

        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            return

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


class _tqdm_cls:
    def __call__(self, *args, **kwargs):
        if _tqdm_active:
            return tqdm_lib.tqdm(*args, **kwargs)
        else:
            return EmptyTqdm(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        self._lock = None
        if _tqdm_active:
            return tqdm_lib.tqdm.set_lock(*args, **kwargs)

    def get_lock(self):
        if _tqdm_active:
            return tqdm_lib.tqdm.get_lock()


tqdm = _tqdm_cls()


def is_progress_bar_enabled() -> bool:
    """Return a boolean indicating whether tqdm progress bars are enabled."""
    global _tqdm_active
    return bool(_tqdm_active)


def enable_progress_bar():
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = True
    hf_hub_utils.enable_progress_bars()


def disable_progress_bar():
    """Disable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = False
    hf_hub_utils.disable_progress_bars()