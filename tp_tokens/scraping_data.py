import re
import unicodedata

TITLE_SYMBOL = ('#', '**', 'ANNEXE')
HTML_TBL_SYMBOL = '<div'
SPECIAL_SYMBOLS = ('(','[','{', '/', ',', '*', '€', '«', ']', '•', '…', ';', '=', '<', '€', '^', '∑', '&', '□', ':')
FORBIDDEN_CHARS = ('∑',"\u0307", "‰", "⁄", "∫", "≤", "˂")

# The idea is to remove [identifier][punctuation][space] that can stand at the beginning of a line
# The identifier may be a digit, a roman literal, or a letter
# Punctuation can be anything from a dash to a parenthesis or a "°"
# Space to separate from the text. 
#
# Such structure might repeat itself (ex : "I. — ") thus the "+" sign at the end
PART_ANNOTATION_RE = re.compile(
    r"""
    ^
    (?:                                         # repeatable annotation blocks
        (?:\d+|[IVXivx]+|[a-zA-Z])?             # identifier
        (?:\ bis\.|\ ter\.|[\.\)\"\-—–―‒‒°]+)   # punctuation (at least one) (the number of different dashes is CRAZY)
        \s*                                     # space
    )+
    """,
    re.VERBOSE,
)

HTML_TAG_RE = re.compile(r"</?em>")

MUTLIPLE_DOTS_RE = re.compile(r"\.\.+")

TRANSLATION_TABLE = str.maketrans({
    # Quotes
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "ˮ": '"',
    "’": "'",
    "‘": "'",
    "ʹ": "'",

    # Dashes
    "–": "-",
    "—": "-",
    "−": "-",
    "‒": "-",
    "●": "-",

    # Spaces
    "\u00A0": " ",   # non-breaking space
    "\u2009": " ",   # thin space
    "\u202F": " ",   # narrow no-break space

    #Special Characters
    ",":',',
    ",":",",
    "i̇,"[1]: "",
})


# helper functions
def normalize_punctuation(s: str) -> str:
    return s.translate(TRANSLATION_TABLE)

def normalize_unicode(s: str) -> str:
    """
    fixing ligatures and "ᵉ", "⁶", "⁹", etc.
    """
    return unicodedata.normalize("NFKC", s)

def strip_annotation(s: str) -> str:
    return PART_ANNOTATION_RE.sub("", s)

def remove_html_tags(s: str) -> str:
    return HTML_TAG_RE.sub("", s)

def remove_repeating_dots(s: str) -> str:
    return MUTLIPLE_DOTS_RE.sub(".", s)

def format_sentence(s: str) -> str:
    out_s = normalize_punctuation(s)
    out_s = normalize_unicode(out_s)
    out_s = strip_annotation(out_s)
    out_s = remove_html_tags(out_s)
    out_s = remove_repeating_dots(out_s)

    out_s = out_s.capitalize()

    return out_s

# Simple checking
def is_a_title(s: str) -> bool:
    return s.startswith(TITLE_SYMBOL)

def is_a_table(s: str) -> bool:
    return s.startswith(HTML_TBL_SYMBOL)

def is_empty(s: str) -> bool:
    return not(bool(s))

def contains_forbidden_char(s: str) -> bool:
    return (
        any(c in FORBIDDEN_CHARS for c in s)
    )

def ignore_str(s: str) -> bool:
    '''
    Filter out edge cases that doesn't improve data quality.
    On every doc of the corpus we loose about 1.000s of sentences out of ~500.000
    '''
    
    return (
        s.startswith(SPECIAL_SYMBOLS)
        or not(s) #empty string
        or contains_forbidden_char(s)
    )



# main procedure
def extract_sentences(filepath : str) -> list[str]:

    lines = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # first 4 lines are useless
            if i < 4:
                continue

            l = line.strip()

            # we filter out unusable material
            if not(
                is_empty(l)
                or is_a_title(l)
                or is_a_table(l)
            ):
                f_seq = format_sentence(l)

                if not(ignore_str(f_seq)):
                    lines.append(f_seq)
    
    return lines
