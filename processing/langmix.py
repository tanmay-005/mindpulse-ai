"""Rough language-mix tagging for reviews.

Phase 0 uses a word list to estimate how much Hinglish the apps receive.
Phase 1 replaces this with a trained language-ID model behind the same function.

Labels: 'english' | 'hinglish' | 'hindi_dev' (mostly Devanagari script) | 'other'
"""
import re
from dataclasses import dataclass

# a "token" is a run of English letters OR a run of Devanagari letters
TOKEN_RE = re.compile(r"[A-Za-z]+|[\u0900-\u097F]+")
DEVANAGARI_RE = re.compile(r"^[\u0900-\u097F]+$")

# Common Roman-script Hindi words in app reviews.
# Words that are ALSO common English words ("me", "to", "the", "so", "do",
# "main", "bus", "par") are left out on purpose, so English reviews aren't miscounted.
ROMAN_HINDI = frozenset("""
    hai hain ha hu hoon ho hota hoti hote hua hui hue tha thi thay
    nahi nahin nhi nai na mat kabhi
    kya kyu kyun kyon kaise kaisa kaisi kab kaha kahan kon kaun kitna kitne kitni
    kuch koi sab sabhi bahut bohot bhot bht bilkul ekdum zyada jyada kam
    aur ya lekin toh phir fir bhi hi sirf abhi ab tak se ko ka ki ke
    mera meri mere tera teri tere apna apni apne uska uski uske hamara humara
    mujhe mujhko tumhe aapko unko isko usko
    hum tum aap wo woh ye yeh yah vo
    karo karna karne kar kiya kiye kiyaa kardo krdo kr krna diya diye de dena
    raha rahi rahe rha rhi gaya gayi gaye gya gyi liya lena milta milti mila mili
    aaya aayi aaye aya ayi aata aati jata jati jaata
    paisa paise paisay rupay rupaye wapas vapas kat kata kaat
    accha acha achha achhi acchi badiya badhiya bekar bekaar bakwas bakwaas
    ghatiya bura buri galat sahi jaldi der
    bhai bhaiya yaar didi ji
    din raat ghanta ghante mahina hafte saal
    khana khaana dukan dukaan saman samaan
""".split())


@dataclass(frozen=True)
class LangMix:
    label: str
    code_mix_index: float   # share of Hindi words (Roman or Devanagari), 0 to 1
    n_tokens: int


def tag_token(token):
    """'hi_dev' (Devanagari), 'hi_rom' (Roman-script Hindi) or 'en'."""
    if DEVANAGARI_RE.match(token):
        return "hi_dev"
    if token.lower() in ROMAN_HINDI:
        return "hi_rom"
    return "en"


def lang_mix(text):
    tokens = TOKEN_RE.findall(text or "")
    if not tokens:                       # empty or emoji-only
        return LangMix("other", 0.0, 0)

    tags = [tag_token(t) for t in tokens]
    n = len(tags)
    n_dev = tags.count("hi_dev")
    n_rom = tags.count("hi_rom")
    cmi = (n_dev + n_rom) / n

    if n_dev / n >= 0.5:
        label = "hindi_dev"
    elif n_rom >= 2 or (n_rom >= 1 and n <= 4) or cmi >= 0.2:
        label = "hinglish"
    else:
        label = "english"
    return LangMix(label, round(cmi, 3), n)