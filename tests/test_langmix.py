from processing.langmix import lang_mix, roman_hindi, tag_token


def test_english_review():
    assert lang_mix("The app crashes every time I open the payment page").label == "english"


def test_hinglish_review():
    m = lang_mix("bhai refund 5 din se nahi aaya, paisa kat gaya order cancel")
    assert m.label == "hinglish"
    assert 0.3 < m.code_mix_index < 0.9


def test_short_hinglish():
    assert lang_mix("bakwas app").label == "hinglish"


def test_devanagari_review():
    assert lang_mix("ऐप बहुत धीमा है, पैसे कट गए").label == "hindi_dev"


def test_mixed_script_mostly_devanagari():
    assert lang_mix("bahut bura hai कस्टमर केयर सर्विस बहुत गंदी है सबसे बढ़िया").label == "hindi_dev"


def test_english_lookalikes_not_counted():
    assert lang_mix("Please do send me the update to fix this").label == "english"


def test_empty_and_emoji_only():
    assert lang_mix("").label == "other"
    assert lang_mix("😡 [angry]").label == "english"   # the tag word counts as English


def test_word_list_loaded_from_file():
    words = roman_hindi()
    assert "nahi" in words and "me" not in words and "to" not in words
    assert tag_token("Nahi") == "hi_rom" and tag_token("पैसे") == "hi_dev" and tag_token("refund") == "en"