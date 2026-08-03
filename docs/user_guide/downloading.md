# Downloading datasets

The public NILM corpora are large in a way that shapes how you work with them.
BLOND is 8.9 TB. UK-DALE's 16 kHz mains recording is published one hour per file
at roughly 200 MB a file, several years deep, and its meter readings come as a
single 3.6 GB archive. HIFDA is one 4.38 GB archive holding 770,612 files; WHITED
is one 2.1 GB archive; PLAID is one 695 MB article. Before you have run a single
experiment, the naive path costs you a week and most of a disk.

Almost none of it is what you wanted. A first look at UK-DALE needs one hour of
waveform and the meter channels covering that hour — a few hundred megabytes out
of terabytes. {mod}`nilmframe.sources` exists to fetch exactly that.

## Two steps, and only one of them costs anything

**Planning** answers *which remote bytes does this configuration need?* It reads
directory listings and archive footers — kilobytes — and returns a
{class}`~nilmframe.sources.Plan` you can print.

**Fetching** takes that plan and materialises it, checking checksums where the
host publishes them, resuming interrupted transfers, and skipping whatever is
already on disk.

## Planning and fetching from Python

Each reader can fetch its own subset and hand you back a reader over it:

```{code-block} python
from nilmframe.readers import UKDALE

reader = UKDALE.download(
    "~/.cache/nilmframe/ukdale",
    houses=[1], channels=[1, 5],
    time_range=(1421784000, 1421870400),   # one day, unix seconds
    max_hf_files=2,
)
```

`UKDALE.plan(...)` takes the same arguments and returns the plan without fetching.
Every reader has the same pair — {class}`~nilmframe.readers.BLOND`,
{class}`~nilmframe.readers.HIFDA`, {class}`~nilmframe.readers.PLAID` and
{class}`~nilmframe.readers.WHITED` — each taking the arguments that dataset is
selected by:

```{code-block} python
from nilmframe.readers import BLOND, HIFDA

blond = BLOND.download("~/.cache/nilmframe/blond",
                       units=["clear", "medal-1"], days=["2016-09-30"],
                       max_seconds=10)
hifda = HIFDA.download("~/.cache/nilmframe/hifda",
                       appliances=["Microwave"], limit=4)
```

## From the command line

The same two steps, without Python:

```{code-block} bash
nilmframe fetch ukdale --cache ~/.cache/nilmframe/ukdale \
                       --channels 1 5 --from 2015-01-20 --to 2015-01-21 \
                       --max-hf-files 2 --dry-run
```

```{code-block} text
ukdale: 5 files, 402.0 MiB (up to 974.7 MiB)
       850 B  low_freq/house_1/labels.dat
  ≤328.4 MiB  low_freq/house_1/channel_1.dat
  ≤244.2 MiB  low_freq/house_1/channel_5.dat
  202.1 MiB  high_freq/house_1/2015/wk04/vi-1421784000_865334.flac
  199.9 MiB  high_freq/house_1/2015/wk04/vi-1421787600_708937.flac
```

Drop `--dry-run` and it fetches. Above a gigabyte it asks first.

The `≤` is honest rather than coy: a meter channel is extracted until the readings
pass the end of your window, and where that lands cannot be known without reading
it. The plan promises an upper bound.

## The cache is an ordinary directory

It is laid out exactly as each reader's documented on-disk layout, so nothing is
hidden:

```{code-block} text
~/.cache/nilmframe/ukdale/
  nilmframe-cache.json                 where each file came from, and its checksum
  low_freq/house_1/labels.dat
  low_freq/house_1/channel_1.dat
  high_freq/house_1/2015/wk04/vi-1421784000_865334.flac
```

Which means two things. If you already have UK-DALE from somewhere else, point a
reader at your copy and never call any of this. And when a fetch goes wrong, you
debug it with `ls`.

`nilmframe-cache.json` records the URL and checksum behind every file. That is
what makes a second run cheap — it skips what is already correct — and what keeps
a store built from the cache traceable to what was actually downloaded.

## What each dataset allows

All six support partial fetches, though none of them advertise it. For what each
one *contains* — resolution, whether it has an aggregate, and the caveats — see
{doc}`datasets_tour`.

**BLOND** is the one that makes the case. Uncompressed it is about 8.9 TB: 213
days of a three-phase office circuit at 50 kHz with fifteen units of separately
metered sockets, plus a 250 kHz release. One day of the whole rig is 42 GB and a
single five-minute mains file is 118 MB. Every file's start time is in its name,
so a day or an hour resolves to a file list from directory listings alone.

The transport is FTP rather than HTTP. mediaTUM's web front end is rate-limited
against crawlers; what it publishes for bulk access is a delivery server with a
per-dataset account, and that is the channel used here — the same one the
dataset's own download tooling uses. `MLSD` reports each file's size in the
listing, so the plan is costed without touching one, and `REST` resumes a
transfer that dropped partway.

```{code-block} bash
nilmframe fetch blond --cache ~/.cache/nilmframe/blond \
                      --units clear medal-1 --days 2016-09-30 --max-files 1
```

That is 142 MB: one five-minute window of three-phase mains at 50 kHz, one
fifteen-minute window of six metered sockets at 6.4 kHz, and the appliance log.

:::{admonition} BLOND's labels are a history, not a snapshot
:class: warning

`appliance_log.json` records each MEDAL's socket configuration *stamped with the
moment it took effect*, because over seven months people unplugged things. The
reader takes the newest configuration at or before each recording's own clock.
Reading the first entry instead silently mislabels whole weeks — and it looks
fine, because every label is a real appliance name.
:::

**HIFDA** is 100 kSPS steady-state voltage and current for fourteen appliances
plus the empty grid — a background class most submetered corpora lack. It lives
on Zenodo, which is the case everything else should look like: a DOI, immutable
versions, an MD5 from the API, and range-supporting storage.

Its one 4.38 GB archive holds 770,612 files, because the same recordings are
published windowed four ways, from 10.24 ms slices to whole activations. Choose
with `--window`; the default takes the 750 whole recordings rather than the
358,740 slices of them. Reading that archive's directory is the one real cost —
770,612 entries is 85 MB of footer — so it is kept on disk and every plan after
the first is free.

```{code-block} bash
nilmframe fetch hifda --cache ~/.cache/nilmframe/hifda \
                      --appliances Microwave --limit 4
```

:::{admonition} HIFDA has no grid fundamental
:class: warning

Its voltage channel is band-limited to 300 Hz – 50 kHz and its current to roughly
30 Hz – 50 kHz, so the 50 Hz component is filtered out of both. The authors were
after the high-frequency signature, but it means cycle alignment has no zero
crossings to find and `v * i` is not the appliance's power. Use it for
high-frequency representations, not for disaggregation against an aggregate.

The files also hold the converter's raw 0–3.3 V output rather than volts and
amperes; the reader applies the release's documented conversion, so the store
gets physical units like every other corpus.
:::

**SmartNIALMeter** is the coarse one. It is on Zenodo, but published as `.7z`,
whose members are grouped into solid blocks and compressed together — so unlike a
zip, one member is not reachable on its own. This release is twelve blocks, which
makes the unit of laziness a block: one building's aggregate costs the roughly
0.85 GB block holding it, against 10 GB for the release. Ask for a whole building
at once; the second file in a block is free once the first has paid for it.

```{code-block} bash
nilmframe fetch smartnialm --cache ~/.cache/nilmframe/snm \
                           --buildings 1 --appliances boiler freezer
```

**UK-DALE waveforms** are genuinely per-hour. The recording's start time is *in
the filename*, which is the same fact the reader already uses to put a waveform
and the meter readings on one clock, so a time range resolves to a file list
without downloading anything. CEDA's directory listings carry each file's size
and MD5, so the plan knows the bill and every download is verified on arrival.

**UK-DALE meter readings** are published only as that one 3.6 GB archive — there
are no per-channel files to link to. They are reached by reading the archive's
directory over a range request and pulling single members out of it. One channel
is about 50 MB compressed against 3.6 GB for the archive, and a `time_range`
drops it further because the extraction stops decompressing once the readings
pass the end of your window. A one-day request in early 2013 costs 13 MB.

**PLAID** is the tidiest: figshare, a DOI, a versioned article, an MD5 from the
API. It is also unexpectedly granular, because its archive stores members
uncompressed — so the waveform archive nested inside it is itself addressable,
and ten recordings cost a couple of megabytes instead of 695 MB.

**WHITED** is one archive on Google Drive, but Drive honours range requests and
every recording's appliance, brand, region and measurement kit are encoded in its
filename. So `--appliances Kettle` is 20 MB out of 2.1 GB. Drive gates large files
behind a virus-scan interstitial, which the fetcher answers; when Drive refuses
for its own reasons, you get Drive's own words back.

Only the 1339 files directly under `DATEN/` are fetched. The archive also holds
`Experiments/` — including runs with two appliances on at once — plus `notUsed/`
and `MIXED/`, whose filenames look like corpus recordings but do not describe one
appliance. The plan says how many it left behind.

:::{admonition} WHITED's password
:class: note

The distribution page publishes a ZIP password. The archive currently served
carries no encrypted members, so nothing here needs it. If that changes, the
fetch fails with a pointer to that page — which publishes both the password and
the citation the authors ask for, and reading it is what the gate is for.
:::

## Fetching high-frequency data alone is a trap

A UK-DALE waveform recording takes its session — and therefore its access to the
submeters that label it — from the low-frequency run that contains it. Fetch
waveforms without the meter channels covering the same hours and every window
lands in the `hf_only` session with nothing to learn from.

So asking for high-frequency data here implies the meter channels over the same
range unless you pass `--no-low-freq`, and if you do, the plan says what you are
giving up.

## Then convert

Fetching gives you vendor files; {doc}`data_loading` turns them into a store. The
fetch prints the exact command, because it knows where everything landed:

```{code-block} text
next: nilmframe convert ukdale --src ~/.cache/nilmframe/ukdale/low_freq \
                               --high-freq-root ~/.cache/nilmframe/ukdale/high_freq \
                               --dst stores/ukdale
```
