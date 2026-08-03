"""Where the datasets live, in one file.

These URLs are the part of the fetcher most likely to break, and the least
interesting to review: UK-DALE moved from ``data.ukedc.rl.ac.uk`` to CEDA and the
old host now answers a redirect to a landing page, which is exactly the kind of
change that should be a one-line edit rather than a hunt through parsing code.

Everything here was checked against the live hosts. Where a number is recorded --
an archive length, a figshare article id -- it is recorded because knowing it up
front saves a request, and each is verified against the host before use, so a
stale constant degrades to a slower plan rather than a wrong one.
"""

from __future__ import annotations

__all__ = [
    "BLOND",
    "FIRED",
    "HIFDA",
    "PLAID",
    "REFIT",
    "SMARTNIALM",
    "UCI",
    "UKDALE",
    "WHITED",
]

#: CEDA serves the UK-DALE archive. Every directory answers a machine-readable
#: listing at ``<url>?json`` carrying, per file, its size and MD5 -- which is a
#: ready-made manifest, so the high-frequency planner never has to guess.
UKDALE = {
    "listing_suffix": "?json",
    "high_freq": (
        "https://data.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption"
        "/Domestic/UK-DALE-2017/UK-DALE-2017-16kHz"
    ),
    # The meter readings are published only as this one archive: there are no
    # per-house or per-channel files to fetch. It is range-readable, so the
    # planner reads its directory and pulls single channels out of it.
    "low_freq_zip": (
        "https://dap.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption"
        "/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.zip"
    ),
    "home": (
        "https://data.ceda.ac.uk/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017"
    ),
}

#: PLAID is on figshare, which is the well-behaved case: a DOI, a versioned
#: article, and an MD5 from the API. The article holds one zip whose members are
#: *stored* rather than deflated, so the waveform archive nested inside it can be
#: read at an offset without unpacking anything.
PLAID = {
    "article": 11605215,
    "api": "https://api.figshare.com/v2/articles/{article}",
    "doi": "10.6084/m9.figshare.11605215.v1",
    # figshare's metadata API refuses some hosts outright -- a datacenter address
    # gets 403 whatever it sends -- while the download host serves the same file
    # normally. These are what the API would have answered, recorded so a machine
    # it dislikes can still fetch. Verified against the API from a host it allows.
    "fallback": {
        "download_url": "https://ndownloader.figshare.com/files/21003861",
        "size": 695078457,
        "md5": "ee15998c3241933d81bea6ad5c3cb059",
    },
    "inner_archive": "2017.zip",
    # Both halves, because they partition the release rather than duplicate it:
    # 719 recordings are described in one and 1074 in the other, with no overlap
    # and nothing left over. Fetching only one silently loses 60% of the corpus.
    "metadata_members": ("meta_2017.json", "meta_2014.json"),
    "home": "https://figshare.com/articles/dataset/PLAID_2017/11605215",
}

#: BLOND is served from mediaTUM's delivery server over FTP, with a per-dataset
#: account whose name is the dataset's own node id. The web front end is
#: rate-limited against crawlers; this is the channel published for bulk access,
#: and it is what the dataset's own download tooling uses.
BLOND = {
    "host": "138.246.224.34",
    "root": "/FD_Share_Kriechbaumer/BLOND",
    "user": "m1375836",
    "password": "m1375836",
    "appliance_log": "appliance_log.json",
    "home": "https://mediatum.ub.tum.de/1375836",
}

#: HIFDA is on Zenodo: a DOI, a versioned record, an MD5 from the API, and
#: storage that honours range requests. Its one archive holds 770,612 members --
#: the same recordings windowed four ways -- so choosing a window matters more
#: here than choosing a file.
HIFDA = {
    "record": 14886758,
    "archive": "HIFDA_HF_electrical_signals_dataset.zip",
    "home": "https://zenodo.org/records/14886758",
}

#: SmartNIALMeter is on Zenodo but published as .7z, whose solid blocks make the
#: unit of laziness a block rather than a file -- see
#: :mod:`nilmframe.sources._sevenzip`.
SMARTNIALM = {
    "record": 10875988,
    "home": "https://zenodo.org/records/10875988",
}

#: FIRED is served from its authors' rsync daemon at Freiburg. The password is
#: printed in the dataset's own README, which is how they intend bulk access to
#: work -- the three commands documented there differ only in what they exclude.
FIRED = {
    "url": "rsync://FIRED@clu.informatik.uni-freiburg.de/FIRED",
    "password": "nobodyGetsFIRED",
    "home": "https://github.com/voelkerb/FIRED_dataset_helper",
}

#: REFIT is on Zenodo as one cleaned CSV per house. This record carries six of
#: the twenty homes; the full release is in an institutional repository that is
#: figshare-backed and refuses some hosts.
REFIT = {
    "record": 5063428,
    "home": "https://zenodo.org/records/5063428",
}

#: The UCI household set: one zip, one text file, and the most-benchmarked
#: low-frequency corpus in the field.
UCI = {
    "url": (
        "https://archive.ics.uci.edu/static/public/235/"
        "individual+household+electric+power+consumption.zip"
    ),
    "archive": "household_power_consumption.zip",
    "member": "household_power_consumption.txt",
    # The host sends chunked with no Content-Length, so the plan cannot be costed
    # from a HEAD; this is the published size, checked against a full fetch.
    "size": 20640916,
    "home": "https://archive.ics.uci.edu/dataset/235",
}

#: WHITED is a single archive on Google Drive. Drive honours range requests, so
#: members are reachable individually, but it gates large files behind a
#: virus-scan interstitial that has to be answered first -- see
#: :func:`nilmframe.sources._http.resolve`.
WHITED = {
    "drive_id": "1HdDDTUmD7p5GjOsVKF9kmh0gcBBX117q",
    "url": "https://drive.google.com/uc?export=download&id=1HdDDTUmD7p5GjOsVKF9kmh0gcBBX117q",
    "member_prefix": "DATEN/",
    "home": "https://www.cs.cit.tum.de/dis/resources/whited/",
}
