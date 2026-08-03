# Signature database

A store of *one representative high-frequency signature per appliance*, drawn from
every corpus that has waveforms, plus a self-contained page for looking at them.

The point is comparison. Nine corpora record the same physical thing at rates from
2 kHz to 100 kSPS, in five container formats, with different conventions for
polarity and units. Once each is cut to one signature and written into a single
store, an appliance from FIRED and the same class from WHITED can be put side by
side without any of that mattering.

Two steps, and they run in that order:

    python tools/signatures/build_store.py   ~/nf-work/stores/signatures
    python tools/signatures/build_page.py    ~/nf-work/stores/signatures  out.html

`build_store.py` reads whatever corpora are cached locally, picks one signature per
appliance per corpus, and writes them as ordinary channels. `build_page.py` reads
that store and emits one HTML file with the data inlined -- no server, no CDN.

## What "one signature" means

It depends on what the corpus records, and the difference is not cosmetic:

**Continuous corpora** (FIRED, BLOND) meter an appliance for weeks. A signature
here is one *run*: switch-on to switch-off, found from the low-rate summary, capped
at an hour. A run is only taken if it sits inside a file -- one touching a boundary
may be truncated, and a truncated run is not a cycle.

**Activation corpora** (WHITED, PLAID, HIFDA) record a few seconds around a single
switch-on. There the whole recording *is* the signature, and the envelope is
correspondingly short. Nothing is padded to make it look otherwise.

Run these on a machine that already has the corpora fetched. Fetching is a separate
concern -- see the downloading chapter.
