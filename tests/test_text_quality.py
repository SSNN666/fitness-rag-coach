"""text_quality 单测：OCR 乱码识别与页面过滤。"""
from text_quality import is_garbled, filter_ocr_pages


def test_garbled_repetition():
    assert is_garbled("嗒嗒嗒嗒嗒嗒嗒嗒嗒嗒嗒嗒") is True


def test_garbled_invalid_chars():
    assert is_garbled("?" * 15) is True


def test_too_short_dropped():
    assert is_garbled("abc") is True


def test_normal_fitness_text_passes():
    assert is_garbled("深蹲主要锻炼股四头肌和臀大肌，注意腰背挺直避免受伤。") is False


def test_filter_split_ok_and_garbled():
    ok, bad = filter_ocr_pages([
        ("p1.png", "深蹲主要锻炼股四头肌和臀大肌，注意腰背挺直避免受伤。", 1),
        ("p2.png", "?????", 2),
        ("p3.png", "硬拉是背部训练的王牌动作，注意腰背挺直。", 3),
    ])
    assert [p[2] for p in ok] == [1, 3]
    assert [p[2] for p in bad] == [2]
