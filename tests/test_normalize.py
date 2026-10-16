import pytest

from processing.normalize import (normalize, shorten_elongations, strip_personal_info,
                                  tag_emojis, unify_spellings)


# ── the full pipeline ──
def test_main_example():
    assert normalize("bhaiii refnd nhi aaya 😡😡") == "bhai refund nahi aaya 😡😡 [angry]"


def test_empty_and_none():
    assert normalize("") == ""
    assert normalize(None) == ""


def test_clean_english_unchanged():
    assert normalize("The app crashes on the payment page.") == "The app crashes on the payment page."


# ── personal info ──
@pytest.mark.parametrize("raw, expected", [
    ("call me 9876543210 now", "call me <phone> now"),
    ("call +91 98765 43210", "call <phone>"),
    ("mail me at abc.def@gmail.com", "mail me at <email>"),
    ("paid to rahul123@okaxis", "paid to <upi>"),
    ("see https://bit.ly/abc for proof", "see <url> for proof"),
    ("txn id 123456789012 failed", "txn id <num> failed"),
])
def test_personal_info_removed(raw, expected):
    assert normalize(raw) == expected


def test_money_amounts_kept():
    assert normalize("₹1500 kat gaye") == "₹1500 kat gaye"
    assert normalize("refund of 2000 rs") == "refund of 2000 rs"


# ── emojis ──
def test_emoji_run_capped_and_tagged():
    assert normalize("worst 😡😡😡😡") == "worst 😡😡 [angry]"


def test_heart_with_variation_selector():
    assert normalize("love it ❤️") == "love it ❤ [love]"


def test_skin_tone_thumbs_up():
    assert normalize("great 👍🏽") == "great 👍 [thumbs_up]"


def test_mixed_emojis():
    assert normalize("😂😂 bakwas 👎") == "😂😂 [laughing] bakwas 👎 [thumbs_down]"


def test_unknown_emoji_left_alone():
    assert tag_emojis("ok 🦄").split() == ["ok", "🦄"]


# ── elongations ──
def test_elongation_letters_and_punctuation():
    assert shorten_elongations("bakwaaaas!!!!") == "bakwaas!!"


def test_elongation_does_not_touch_digits():
    assert shorten_elongations("paid 1000 rs") == "paid 1000 rs"


# ── spellings ──
@pytest.mark.parametrize("raw, expected", [
    ("nhi", "nahi"), ("Nahin", "nahi"), ("plz", "please"), ("bcoz", "because"),
    ("bhot", "bahut"), ("acha", "accha"), ("app bakwas h", "app bakwas hai"),
])
def test_spelling_variants(raw, expected):
    assert unify_spellings(raw) == expected


def test_elongated_slang_word():
    assert normalize("nahiiii yaaar") == "nahi yaar"


def test_unknown_words_kept_with_case():
    assert unify_spellings("Zomato WORST") == "Zomato WORST"


def test_whole_words_only():
    # "h" maps to "hai", but must not change the h inside other words
    assert unify_spellings("hello hi") == "hello hi"


def test_aap_is_hindi_you_not_app():
    assert normalize("aap ne mera paisa wapas nahi kiya") == "aap ne mera paisa wapas nahi kiya"