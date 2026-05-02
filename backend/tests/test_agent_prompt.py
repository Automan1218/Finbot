from app.agent.prompt import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    build_prompt,
    prompt_prefix_token_estimate,
)


def test_system_prompt_is_stable_string():
    assert isinstance(SYSTEM_PROMPT, str)
    assert SYSTEM_PROMPT == SYSTEM_PROMPT


def test_few_shot_examples_is_list_of_role_content_dicts():
    assert isinstance(FEW_SHOT_EXAMPLES, list)
    assert len(FEW_SHOT_EXAMPLES) >= 4
    for item in FEW_SHOT_EXAMPLES:
        assert set(item.keys()) == {"role", "content"}
        assert item["role"] in {"user", "assistant", "system"}


def test_prefix_token_estimate_meets_openai_cache_threshold():
    assert prompt_prefix_token_estimate() >= 1024


def test_build_prompt_places_static_prefix_first():
    history = [{"role": "user", "content": "prev question"}]
    rag = "policy snippet"
    user_msg = "current question"

    messages = build_prompt(history=history, rag_context=rag, user_msg=user_msg)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    for i, shot in enumerate(FEW_SHOT_EXAMPLES, start=1):
        assert messages[i] == shot
    assert messages[-1] == {"role": "user", "content": "current question"}
    rag_position = 1 + len(FEW_SHOT_EXAMPLES)
    assert "policy snippet" in messages[rag_position]["content"]


def test_build_prompt_handles_empty_rag_and_history():
    messages = build_prompt(history=[], rag_context="", user_msg="hello")

    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[-1] == {"role": "user", "content": "hello"}
    rag_position = 1 + len(FEW_SHOT_EXAMPLES)
    assert messages[rag_position]["role"] == "system"


def test_build_prompt_static_prefix_byte_identical_across_calls():
    p1 = build_prompt(history=[], rag_context="x", user_msg="a")
    p2 = build_prompt(
        history=[{"role": "user", "content": "y"}],
        rag_context="z",
        user_msg="b",
    )
    prefix_len = 1 + len(FEW_SHOT_EXAMPLES)
    assert p1[:prefix_len] == p2[:prefix_len]


def test_build_prompt_strips_session_metadata_from_history():
    messages = build_prompt(
        history=[{"role": "assistant", "content": "done", "created_at": "x"}],
        rag_context="",
        user_msg="next",
    )

    assert {"role": "assistant", "content": "done"} in messages
