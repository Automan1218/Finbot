from app.cache.normalize import make_llm_cache_key, normalize_query


def test_normalize_strips_punctuation_case_filler():
    assert normalize_query("帮我查餐饮！") == "查餐饮"
    assert normalize_query("请告诉我餐饮花了多少") == "餐饮花了多少"
    assert normalize_query("  Hello,  World!  ") == "hello world"


def test_normalize_collapses_whitespace():
    assert normalize_query("a   b\n c\t d") == "a b c d"


def test_normalize_returns_empty_for_only_filler():
    assert normalize_query("帮我请一下") == ""


def test_make_llm_cache_key_stable_for_synonymous_inputs():
    k1 = make_llm_cache_key("帮我查餐饮", "team-1", "2026-05")
    k2 = make_llm_cache_key("请查一下餐饮", "team-1", "2026-05")
    k3 = make_llm_cache_key("查餐饮", "team-1", "2026-05")
    assert k1 == k2 == k3
    assert k1.startswith("llm:resp:")


def test_make_llm_cache_key_differs_by_team_or_month():
    base = make_llm_cache_key("查餐饮", "team-1", "2026-05")
    assert base != make_llm_cache_key("查餐饮", "team-2", "2026-05")
    assert base != make_llm_cache_key("查餐饮", "team-1", "2026-06")
