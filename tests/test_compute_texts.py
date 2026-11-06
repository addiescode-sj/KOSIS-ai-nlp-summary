"""kosis_chat/compute/texts.py (T1-28 S2b, D-c (a)): the ten f-string refusals of resolve.py / semantics.py, as
functions. Each expected string is the f-string that stood at the call site, written out for the sample arguments."""

from __future__ import annotations

from kosis_chat.compute import texts


def test_resolve_sentences():
    assert texts.event_unknown("코로나19") == "'코로나19'이(가) 언제인지 알 수 없습니다. 연도를 알려 주세요."
    assert texts.path_ambiguous(["부산광역시", "중구"]) == "'부산광역시 > 중구'이(가) 여러 곳입니다. 어느 것인지 골라 주세요."
    assert texts.pick_category("행정구역별") == "행정구역별을(를) 골라 주세요."
    assert texts.span_not_supported("2024", "H1") == "2024년 H1: 분기·반기 표의 시점 해석은 아직 지원하지 않습니다."
    assert texts.span_on_yearly("2024", "Q2") == "이 표는 연 단위라 2024년 Q2 값이 없습니다."
    assert texts.other_table("DT_X", "DT_1B8000F") == "DT_X: 이 질문의 표(DT_1B8000F)가 아닙니다(여러 표 결합은 아직 엔진만 지원)."
    assert texts.star_outside_map("시도별") == "시도별의 '*'는 map 안에서만 쓸 수 있습니다."


def test_semantics_sentences():
    assert texts.pp_diff_units("서울·부산", "명", "%") == "서울·부산: %p 차이는 두 값이 모두 %일 때만 구할 수 있습니다(단위 명 / %)."
    assert texts.pp_diff_units("x", None, "%") == "x: %p 차이는 두 값이 모두 %일 때만 구할 수 있습니다(단위 None / %)."
    assert (texts.share_outside_total(["시점 2024≠2023", "항목 a⊄b"])
            == "비중의 분모가 분자를 포함하지 않습니다(시점 2024≠2023, 항목 a⊄b).")
    assert texts.share_out_of_bounds("서울", 123.456) == "서울 비중이 123.5%로 0~100% 밖이라 분모가 분자의 전체가 아닙니다."
    assert texts.share_out_of_bounds("서울", -0.04) == "서울 비중이 -0.0%로 0~100% 밖이라 분모가 분자의 전체가 아닙니다."


def test_same_as_the_old_f_strings():
    """The f-strings as they stood in resolve.py / semantics.py, evaluated on the same arguments."""
    name, path, axis, year, part, table, tbl_id = "IMF", ["a", "b"], "성별", 2025, "year", "T9", "T1"
    what, a, b, wrong, value = "합계", "%", None, ["x"], 101.0
    assert texts.event_unknown(name) == f"'{name}'이(가) 언제인지 알 수 없습니다. 연도를 알려 주세요."
    assert texts.path_ambiguous(path) == f"'{' > '.join(path)}'이(가) 여러 곳입니다. 어느 것인지 골라 주세요."
    assert texts.pick_category(axis) == f"{axis}을(를) 골라 주세요."
    assert texts.span_not_supported(year, part) == f"{year}년 {part}: 분기·반기 표의 시점 해석은 아직 지원하지 않습니다."
    assert texts.span_on_yearly(year, part) == f"이 표는 연 단위라 {year}년 {part} 값이 없습니다."
    assert texts.other_table(table, tbl_id) == f"{table}: 이 질문의 표({tbl_id})가 아닙니다(여러 표 결합은 아직 엔진만 지원)."
    assert texts.star_outside_map(axis) == f"{axis}의 '*'는 map 안에서만 쓸 수 있습니다."
    assert texts.pp_diff_units(what, a, b) == f"{what}: %p 차이는 두 값이 모두 %일 때만 구할 수 있습니다(단위 {a} / {b})."
    assert texts.share_outside_total(wrong) == f"비중의 분모가 분자를 포함하지 않습니다({', '.join(wrong)})."
    assert texts.share_out_of_bounds(what, value) == f"{what} 비중이 {value:.1f}%로 0~100% 밖이라 분모가 분자의 전체가 아닙니다."
