from app.core.text import (
    approx_token_count,
    lexical_terms,
    normalize_text,
    query_terms,
    sentence_spans,
    split_sentences,
)


def test_normalize_text_unifies_whitespace_and_unicode():
    raw = "Hello\r\nworld !\t\x07\n\n\n\nNext   \nline"
    out = normalize_text(raw)
    assert "\r" not in out and "\x07" not in out and "\t" not in out
    assert "\n\n\n" not in out
    assert out.startswith("Hello\nworld !")


def test_lexical_terms_stems_and_drops_stopwords():
    terms = lexical_terms("The engineers are running the indexes")
    assert "the" not in terms and "are" not in terms
    assert "engin" in terms and "run" in terms and "index" in terms


def test_lexical_terms_keeps_compound_identifiers():
    terms = lexical_terms("Error QL-5031 in retrieval.hybrid_alpha")
    assert "ql-5031" in terms and "ql" in terms and "5031" in terms
    assert "retrieval.hybrid_alpha" in terms


def test_query_terms_are_unique_and_ordered():
    assert query_terms("index Index INDEX search") == ["index", "search"]


def test_split_sentences_protects_initials_and_titles():
    text = "A. Chen met Dr. Okoye at Quillon Labs Inc. yesterday. They agreed on a plan! Next steps follow."
    sentences = split_sentences(text)
    assert sentences[0].startswith("A. Chen met Dr. Okoye at Quillon Labs Inc. yesterday.")
    assert len(sentences) == 3


def test_split_sentences_keeps_list_items_separate():
    text = "Action items\n- Priya: estimate cost\n- Jonas: build dashboard"
    assert split_sentences(text) == ["Action items", "- Priya: estimate cost", "- Jonas: build dashboard"]


def test_sentence_spans_index_original_text():
    text = "First sentence here. Second one\nwraps lines. Third."
    for start, end in sentence_spans(text):
        assert text[start:end].strip() == text[start:end]
        assert start < end
    assert text[slice(*sentence_spans(text)[0])] == "First sentence here."


def test_approx_token_count_counts_words_and_punctuation():
    assert approx_token_count("Hello, world!") == 4
    assert approx_token_count("") == 0


def test_short_lines_are_not_merged_but_wrapped_prose_is():
    agenda = "09:00 Keynote - Hannah Brooks\n11:00 Halcyon roadmap - Alice Chen"
    assert split_sentences(agenda) == ["09:00 Keynote - Hannah Brooks", "11:00 Halcyon roadmap - Alice Chen"]
    wrapped = "This is a long hard-wrapped paragraph line that goes past sixty chars\nand continues here."
    assert split_sentences(wrapped) == [wrapped.replace("\n", " ")]


def test_agenda_lines_with_long_items_are_not_chained():
    agenda = (
        '09:00 Keynote: "Search that knows when it does not know" - Hannah Brooks\n'
        "11:00 Halcyon roadmap for 2026 - Alice Chen\n"
        "16:30 Walking tour of the Alfama district"
    )
    assert len(split_sentences(agenda)) == 3
