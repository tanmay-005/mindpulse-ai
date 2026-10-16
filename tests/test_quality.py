import pytest

from processing.quality import find_near_duplicates, is_spam


@pytest.mark.parametrize("text", [
    "", "   ", "....", "1234", "hhhhhhhhhh", "aaaaaaaab",
    "Use my referral code ABC123 and get ₹100", "Refer and earn ₹500 daily",
    "join my telegram channel for free recharge",
    "Referral code: ZEPTO500 use it", "earn money online daily 5000",
])
def test_spam(text):
    assert is_spam(text)


@pytest.mark.parametrize("text", [
    "good", "👍", "nice app", "ok.........", "😡😡 [angry]", "₹1500 kat gaye refund nahi aaya",
    "bahut bekar service", "कस्टमर केयर बहुत खराब है",
    "the coupon did not apply at checkout",          # mentions coupon, but is a real complaint
    "Refund not received after 10 days",
    "i can not showing referral code to share other",
    "my zepto referal link not showing problem please fix this problem",
    "Referral Code",
    "they robotically offer a small promo code and end the chat",
    "ಸೂಪರ್", "சூப்பர்", "ગુડ", "𝙻𝚒𝚝𝚎𝚛𝚊𝚕𝚕𝚢 𝚋𝚎𝚜𝚝 𝚏𝚘𝚛 𝚞𝚜𝚎", "yoohoo", "niceeeeeewe",
    "⭐⭐⭐⭐⭐", "NEVER USE BLINKIT IF YOU WANT TO SAVE YOUR HARD EARN MONEY",
])
def test_not_spam(text):
    assert not is_spam(text)


def test_exact_and_near_copies_found():
    rows = [
        ("a", "Worst app ever, my refund of 500 rupees is still pending after ten days"),
        ("b", "Delivery boy was very rude and the food arrived cold, never ordering again"),
        ("c", "Worst app ever, my refund of 500 rupees is still pending after ten days!!"),   # near copy of a
        ("d", "worst app ever, my refund of 500 rupees is still pending after ten days"),     # exact copy (case)
        ("e", "Great discounts this week and very fast delivery, totally happy with it"),
    ]
    dup = find_near_duplicates(rows)
    assert dup == {"c": "a", "d": "a"}


def test_different_reviews_not_matched():
    rows = [("a", "refund not received for my cancelled order"),
            ("b", "app crashes every time I open the payment page")]
    assert find_near_duplicates(rows) == {}