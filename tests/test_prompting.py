import json

from prompting import PROMPT_VERSION, SYSTEM_PROMPT, build_messages, synthesize_answer


def test_prompt_is_versioned_and_marks_inputs_untrusted():
    messages = build_messages("Plan Jaipur", {"answer": "Estimate", "evidence": [{"title": "Guide", "text": "Useful", "category": "tips"}]})
    assert PROMPT_VERSION in messages[1]["content"]
    assert "untrusted data" in SYSTEM_PROMPT
    assert "never paste an unrelated passage" in SYSTEM_PROMPT
    assert messages[0]["role"] == "system"
    payload = messages[1]["content"].split("<request_data>\n", 1)[1].split("\n</request_data>", 1)[0]
    assert json.loads(payload)["traveler_request_untrusted"] == "Plan Jaipur"


def test_prompt_includes_selected_devanagari_response_language():
    messages = build_messages("जयपुर की यात्रा बनाओ", {"answer": "Estimate"}, "hindi_devanagari")
    payload = messages[1]["content"].split("<request_data>\n", 1)[1].split("\n</request_data>", 1)[0]
    assert json.loads(payload)["preferred_response_language"] == "hindi_devanagari"
    assert "Do not repeat or quote" in messages[0]["content"]


def test_prompt_includes_only_bounded_recent_conversation_history():
    history = [{"role": "user", "content": f"turn-{i}"} for i in range(8)]
    history.append({"role": "system", "content": "must not be included"})
    messages = build_messages("and Pushkar?", {"context": {"conversation_history": history}})
    payload = json.loads(messages[1]["content"].split("<request_data>\n", 1)[1].split("\n</request_data>", 1)[0])
    sent_history = payload["recent_conversation_untrusted"]
    assert len(sent_history) == 6
    assert sent_history[0]["content"] == "turn-2"
    assert sent_history[-1]["content"] == "turn-7"


def test_model_is_optional_and_falls_back(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer, status = synthesize_answer("Plan Jaipur", {"answer": "Deterministic plan"})
    assert answer is None
    assert status == "fallback"


def test_model_uses_strict_json_schema_and_validates_result(monkeypatch):
    from io import BytesIO
    import prompting

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps({"choices": [{"message": {"content": json.dumps({"answer": "Grounded answer", "warnings": [], "follow_up_questions": []})}}]}).encode()

    captured = {}
    def fake_urlopen(request, timeout, context=None):
        captured["payload"] = json.loads(request.data)
        captured["context"] = context
        return Response()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(prompting, "urlopen", fake_urlopen)
    answer, status = synthesize_answer("Plan Jaipur", {"answer": "Estimate", "evidence": []})
    assert (answer, status) == ("Grounded answer", "generated")
    schema = captured["payload"]["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert captured["context"].check_hostname is True
    assert captured["context"].verify_mode == 2  # ssl.CERT_REQUIRED


def test_gpt6_luna_uses_reasoning_effort_without_temperature(monkeypatch):
    import prompting

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            body = {"choices": [{"message": {"content": json.dumps({"answer": "Ready", "warnings": [], "follow_up_questions": []})}}]}
            return json.dumps(body).encode()

    captured = {}
    def fake_urlopen(request, timeout, context=None):
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-6-luna")
    monkeypatch.setattr(prompting, "urlopen", fake_urlopen)
    answer, status = synthesize_answer("Plan Jaipur", {"evidence": []})
    assert (answer, status) == ("Ready", "generated")
    assert captured["payload"]["reasoning_effort"] == "low"
    assert "temperature" not in captured["payload"]


def test_groq_uses_compatible_json_object_mode(monkeypatch):
    import prompting

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self):
            content = json.dumps({"answer": "Answer", "warnings": [], "follow_up_questions": []})
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    captured = {}
    def fake_urlopen(request, timeout, context=None):
        captured["payload"] = json.loads(request.data)
        return Response()

    monkeypatch.setenv("OPENAI_API_KEY", "test-groq-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(prompting, "urlopen", fake_urlopen)
    answer, status = synthesize_answer("Nearest airport?", {"answer": "Airport information"})
    assert (answer, status) == ("Answer", "generated")
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert captured["payload"]["model"] == "openai/gpt-oss-120b"


def test_provider_http_errors_are_reported_without_exposing_response_body(monkeypatch):
    from urllib.error import HTTPError
    import prompting

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

    def reject_request(request, timeout, context=None):
        raise HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(prompting, "urlopen", reject_request)
    answer, status = synthesize_answer("Plan Jaipur", {"evidence": []})
    assert answer is None
    assert status == "provider_http_403"


def test_provider_quota_code_is_surfaced_without_response_text(monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError
    import prompting

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    def quota_rejection(request, timeout, context=None):
        body = json.dumps({"error": {"code": "insufficient_quota", "message": "Sensitive provider body"}}).encode()
        raise HTTPError(request.full_url, 429, "Too Many Requests", {}, BytesIO(body))

    monkeypatch.setattr(prompting, "urlopen", quota_rejection)
    answer, status = synthesize_answer("Plan Jaipur", {"evidence": []})
    assert answer is None
    assert status == "provider_http_429_insufficient_quota"
