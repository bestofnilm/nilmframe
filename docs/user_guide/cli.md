# Command line

Eight commands, covering the parts of the workflow that are the same every time.

```{code-block} text
nilmframe fetch      download part of a public dataset into a cache
nilmframe convert    convert a source dataset into a canonical store
nilmframe describe   summarise a store
nilmframe compat     report what varies across stores, and what it breaks
nilmframe merge      combine stores under explicit rules
nilmframe train      train one model and write a checkpoint
nilmframe evaluate   score a store's windows without training
nilmframe sweep      run an experiment grid and emit a results table
```

Everything they do is available from Python. They exist because converting a dataset
and running a sweep are the two things you want in a shell script, a Makefile or a job
submission, and neither should require writing a driver.

## fetch

Work out which files a subset of a public dataset needs, then fetch them. See
{doc}`downloading`.

```{code-block} bash
nilmframe fetch ukdale --cache ~/.cache/nilmframe/ukdale \
                       --channels 1 5 --from 2015-01-20 --to 2015-01-21 \
                       --max-hf-files 2 --dry-run
nilmframe fetch plaid  --cache ~/.cache/nilmframe/plaid --limit 20
nilmframe fetch whited --cache ~/.cache/nilmframe/whited --appliances Kettle Fan
nilmframe fetch blond  --cache ~/.cache/nilmframe/blond \
                       --units clear medal-1 --days 2016-09-30 --max-files 1
nilmframe fetch hifda  --cache ~/.cache/nilmframe/hifda --appliances Microwave --limit 4
nilmframe fetch smartnialm --cache ~/.cache/nilmframe/snm --buildings 1
nilmframe fetch fired  --cache ~/.cache/nilmframe/fired --resolution 1Hz
nilmframe fetch refit  --cache ~/.cache/nilmframe/refit --houses 1 2
nilmframe fetch uci    --cache ~/.cache/nilmframe/uci
```

Planning is separate from fetching and costs a few directory listings, so
`--dry-run` prints the bill — file list and total — before anything is spent. Past a
gigabyte the command asks before proceeding; `--yes` answers in advance and
`--max-bytes 20G` refuses outright. Re-running skips whatever is already cached, so
an interrupted fetch is resumed by repeating the command.

`--from`/`--to` take unix seconds or an ISO date read as UTC. For UK-DALE they bound
both halves: which waveform hours to fetch, and where to stop extracting the meter
channels. When it finishes it prints the `convert` command for what it fetched.

## convert

Read a source dataset once and write a canonical store — parquet metadata plus
memory-mapped signal arrays. See {doc}`data_loading`.

```{code-block} bash
nilmframe convert plaid  --src PLAID/CSV --metadata PLAID/meta.json --dst stores/plaid
nilmframe convert whited --src WHITED --dst stores/whited
nilmframe convert ukdale --src ukdale/low_freq --dst stores/ukdale \
                         --rate-hz 0.1667 --houses 1 2 --max-seconds 60 --verify
```

`--max-seconds` bounds the waveform reads only, not the meter channels — a UK-DALE
waveform file is an hour at 16 kHz and you rarely want all of it. `--no-high-freq` skips
them entirely. `--limit` caps the number of recordings, which is how you produce a small
store to develop against.

`--verify` re-hashes everything after writing, which is worth the minute.

## describe

```{code-block} bash
nilmframe describe stores/ukdale --verify
```

Prints the per-appliance table — channels, instances, brands, hours — plus the store's
manifest. `--verify` re-hashes every signal file against the checksum recorded when it
was written, so a truncated download or an interrupted copy is caught here rather than
as a strange training curve.

The columns to read are `instances` and `brands`. See {doc}`data_loading`.

## compat

```{code-block} bash
nilmframe compat stores/ukdale stores/plaid stores/whited
```

Enumerates every axis along which the stores disagree — sampling rate, mains frequency,
supply voltage, quantities, vocabulary — and says which of them actually block each
view. Run it before merging, not after. `--shallow` skips the voltage sampling and reads
metadata only, which is faster on a large store.

{doc}`combining` explains what the axes mean.

## merge

```{code-block} bash
nilmframe merge stores/ukdale stores/plaid --dst stores/combined \
                --require voltage f0 \
                --rename refrigerator=fridge microwave_oven=microwave \
                --normalize-voltage 230
```

Rules are arguments rather than assumptions. `--require` refuses the merge when a named
axis disagrees, and says what differs. `--rename` harmonises the label space.
`--normalize-voltage` brings every channel to one supply level without changing any
load's power — current is scaled inversely.

`--no-prefix` turns off dataset-prefixed channel ids. Only use it if you are certain the
ids are already globally unique.

## train

```{code-block} bash
nilmframe train --store stores/ukdale \
                --view highfreq --align fitps --n-cycles 20 --cycle-size 128 \
                --split leave_house_out --test-size 0.3 --seed 0 \
                --model convnet --width 128 --epochs 20 --augment \
                --out model.pt --accelerator gpu
```

Writes a checkpoint carrying the state dict, the appliance list, the view and the split
manifest — so the file answers "what does column 3 mean" and "what was this trained on"
without anyone having to remember.

## evaluate

```{code-block} bash
nilmframe evaluate --store stores/ukdale --view lowfreq --fold val
```

Scores a store's windows without training. Useful for a sanity check on a new store, and
for measuring how hard a split actually is before spending GPU time on it.

## sweep

```{code-block} bash
nilmframe sweep configs/lf_vs_hf.yaml --store stores/ukdale --out runs/lf_vs_hf
```

Runs every arm of a config against **one shared split**, which is what makes the
comparison a comparison. `--epochs` overrides every arm's epoch count at once, for a
quick smoke run before committing to the real thing. `--progress` shows the training
bars.

`--out` gets `results.csv` and `manifest.json`.

## A whole experiment

```{code-block} bash
nilmframe convert  ukdale --src ukdale/low_freq --dst stores/ukdale --verify
nilmframe describe stores/ukdale
nilmframe compat   stores/ukdale
nilmframe sweep    configs/lf_vs_hf.yaml --store stores/ukdale --out runs/lf_vs_hf
```

Four commands, one config file under version control, one results directory carrying its
own manifest. That is the intended shape: the interesting decisions live in the YAML
where a reviewer can see them, and the shell script contains nothing anybody needs to
argue about.
