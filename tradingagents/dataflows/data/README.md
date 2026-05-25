# Sentiment Lexicons

`sentiment_scoring.py`'s `score_knu()` loads two files at startup and merges
them — overlay values win on key collision:

1. **`knu_sentiment_lexicon.json`** — Full KnuSentiLex (14,852 entries) from
   <https://github.com/park1200656/KnuSentiLex>. General-emotion vocabulary,
   converted to `{word: polarity_int}` form on the `[-2, +2]` polarity scale.
2. **`knu_financial_overlay.json`** — Hand-curated finance-domain overlay
   (~80 entries: 급등 / 폭락 / 호재 / 악재 / 상승 / 하락 / ...). KnuSentiLex
   alone misses these core market terms, so the overlay restores the signal
   on top of the general lexicon.

## Why two files?

KnuSentiLex is a general Korean emotion lexicon and a useful base, but
- it has very little finance vocabulary, and
- many of its entries are phrases ("급하게 몰아쉬는") or inflected forms
  ("급하다"), which our syllable-run tokenizer doesn't match well.

Without the overlay, real ticker discussion ends up with oov_ratio ≳ 90%.
The overlay is small, finance-specific, single-token, and bumps coverage
on retail Korean stock-board text significantly.

## Refreshing the base lexicon

To pull a newer upstream snapshot of KnuSentiLex:

```python
import json
import urllib.request
url = "https://raw.githubusercontent.com/park1200656/KnuSentiLex/master/data/SentiWord_info.json"
src = json.loads(urllib.request.urlopen(url).read())
words = {}
for entry in src:
    w = entry.get("word", "").strip()
    if not w:
        continue
    try:
        words[w] = int(entry["polarity"])
    except (KeyError, ValueError, TypeError):
        continue
out = {
    "_meta": {
        "format": "KnuSentiLex",
        "polarity_range": [-2, 2],
        "source": "https://github.com/park1200656/KnuSentiLex (data/SentiWord_info.json)",
        "count": len(words),
    },
    "words": words,
}
json.dump(out, open("knu_sentiment_lexicon.json", "w"), ensure_ascii=False, indent=2)
```

Then re-run `tests/dataflows/test_sentiment_scoring.py` to confirm the
finance-related cases still pass — they depend on the overlay, not the base.

## Extending the overlay

If a finance term keeps showing up in `oov_ratio` reports during operation,
add it to `knu_financial_overlay.json` with a polarity in `[-2, +2]`. The
overlay is small enough to hand-curate; promote to KoBERT only if even the
overlay can't keep up with the corpus.

## License

KnuSentiLex is distributed under terms that allow research/non-commercial
use; commercial users should consult the upstream repository for terms.
The finance overlay shipped here is hand-curated for this project (no
attribution requirement).
