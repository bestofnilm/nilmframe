"""Tests for the fetching layer.

Everything here runs against a local HTTP server that speaks byte ranges, built
in the fixture below. That is not a compromise: the machinery being tested *is*
range arithmetic and archive parsing, and a local server exercises it end to end
-- footer, directory, local header, inflate, resume -- without depending on a
research council's uptime.

The two archive shapes that only appear in the wild are built by hand:
``_as_zip64`` rewrites a footer the way a >4 GB archive carries it, and
``_central_entry`` builds a directory entry with the overflow sentinels. Both
paths are silently wrong if misread, which is the kind that costs a day.
"""

from __future__ import annotations

import io
import json
import re
import struct
import threading
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nilmframe.sources import Artifact, FetchError, Plan, RemoteZip, fetch, human_bytes
from nilmframe.sources._zip import _zip64_fields
from nilmframe.sources.fetch import _bounded, _Ledger

# --------------------------------------------------------------------------- #
# a server that speaks ranges
# --------------------------------------------------------------------------- #


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:
        store = self.server.store
        name = self.path.lstrip("/").split("?")[0]
        if name not in store:
            self.send_error(404, "no such file")
            return
        body = store[name]

        span = re.fullmatch(r"bytes=(\d+)-(\d*)", self.headers.get("Range", "") or "")
        if span and not self.server.ignore_ranges:
            start = int(span.group(1))
            stop = int(span.group(2)) if span.group(2) else len(body) - 1
            stop = min(stop, len(body) - 1)
            if start >= len(body):
                self.send_error(416, "range not satisfiable")
                return
            chunk = body[start : stop + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{stop}/{len(body)}")
        else:
            chunk = body
            self.send_response(200)
        self.send_header("Content-Length", str(len(chunk)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def host():
    """A local origin serving an in-memory dict of files, with range support."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.store = {}
    server.ignore_ranges = False
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class Host:
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def put(self, name: str, body: bytes) -> str:
            server.store[name] = body
            return f"{self.base}/{name}"

        @staticmethod
        def ignore_ranges(value: bool = True) -> None:
            server.ignore_ranges = value

    yield Host()
    server.shutdown()
    server.server_close()


def _zip_bytes(members: dict[str, bytes], *, stored: set[str] = frozenset()) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, body in members.items():
            method = zipfile.ZIP_STORED if name in stored else zipfile.ZIP_DEFLATED
            archive.writestr(zipfile.ZipInfo(name), body, compress_type=method)
    return buffer.getvalue()


def _as_zip64(blob: bytes) -> bytes:
    """Rewrite a zip footer the way an archive past the 32-bit limits carries it.

    ``zipfile`` only emits these records when the real numbers overflow, and a
    test archive never will, so the shape has to be built to be tested.
    """
    idx = blob.rfind(b"PK\x05\x06")
    count, cd_size, cd_offset = struct.unpack("<HII", blob[idx + 10 : idx + 20])
    z64 = struct.pack(
        "<4sQHHIIQQQQ", b"PK\x06\x06", 44, 45, 45, 0, 0, count, count, cd_size, cd_offset
    )
    locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, idx, 1)
    eocd = struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0)
    return blob[:idx] + z64 + locator + eocd


# --------------------------------------------------------------------------- #
# plan vocabulary
# --------------------------------------------------------------------------- #


def test_artifact_refuses_to_escape_the_cache():
    # Member names come out of an archive, which is untrusted input.
    for escape in ("../secrets", "/etc/passwd", "a/../../b"):
        with pytest.raises(ValueError, match="inside the cache"):
            Artifact("http://x/a.zip", escape)


def test_plan_reports_a_lower_and_an_upper_bound():
    plan = Plan(
        "demo",
        (
            Artifact("http://x/a", "a", size=1000),
            Artifact("http://x/b", "b", size_max=9000),
            Artifact("http://x/c", "c"),
        ),
    )
    assert plan.nbytes == 1000
    assert plan.nbytes_max == 10_000
    assert plan.n_unsized == 1
    assert "up to" in plan.summary()
    assert "≤" in plan.summary()


def test_summary_elides_a_long_plan():
    plan = Plan("demo", tuple(Artifact("http://x/f", f"f{i}", size=1) for i in range(30)))
    assert "and 18 more" in plan.summary(limit=12)


@pytest.mark.parametrize(
    ("count", "expected"), [(0, "0 B"), (999, "999 B"), (1024, "1.0 KiB"), (1 << 40, "1.0 TiB")]
)
def test_human_bytes(count, expected):
    assert human_bytes(count) == expected


# --------------------------------------------------------------------------- #
# reading an archive over ranges
# --------------------------------------------------------------------------- #


def test_remote_zip_reads_members_without_the_archive(host):
    members = {"house_1/labels.dat": b"1 kettle\n2 fridge\n", "house_1/channel_1.dat": b"x" * 5000}
    url = host.put("ukdale.zip", _zip_bytes(members))
    archive = RemoteZip(url)

    assert set(archive.entries) == set(members)
    assert archive.entries["house_1/channel_1.dat"].size == 5000
    for name, body in members.items():
        assert b"".join(archive.stream(name)) == body


def test_remote_zip_reads_a_stored_member(host):
    url = host.put("s.zip", _zip_bytes({"raw.bin": b"\x00\x01\x02" * 100}, stored={"raw.bin"}))
    archive = RemoteZip(url)
    assert archive.entries["raw.bin"].method == 0
    assert b"".join(archive.stream("raw.bin")) == b"\x00\x01\x02" * 100


def test_remote_zip_reads_a_zip64_footer(host):
    """The branch UK-DALE's 3.6 GB archive takes, built by hand."""
    url = host.put("big.zip", _as_zip64(_zip_bytes({"a.txt": b"hello", "b.txt": b"world"})))
    archive = RemoteZip(url)
    assert sorted(archive.entries) == ["a.txt", "b.txt"]
    assert b"".join(archive.stream("b.txt")) == b"world"


def test_directory_entry_with_overflow_sentinels():
    """A member whose real size only appears in its ZIP64 extra field."""
    name = b"mains.dat"
    extra = struct.pack("<HHQQ", 0x0001, 16, 5_000_000_000, 1_200_000_000)
    entry = (
        b"PK\x01\x02"
        + struct.pack("<HHHH", 45, 45, 0, 8)
        + struct.pack("<HHI", 0, 0, 0)  # time, date, crc
        + struct.pack("<II", 0xFFFFFFFF, 0xFFFFFFFF)  # sizes: both overflowed
        + struct.pack("<HHHHHII", len(name), len(extra), 0, 0, 0, 0, 500)
        + name
        + extra
    )
    parsed = RemoteZip("http://x/z.zip")._parse_directory(entry, 1)
    assert parsed["mains.dat"].size == 5_000_000_000
    assert parsed["mains.dat"].compressed_size == 1_200_000_000
    assert parsed["mains.dat"].header_offset == 500


def test_zip64_fields_returns_only_what_overflowed():
    blob = struct.pack("<HHQQ", 0x0001, 16, 11, 22)
    assert _zip64_fields(blob, 2) == [11, 22]
    assert _zip64_fields(b"", 2) == []


def test_nested_archive_is_read_in_place(host):
    """PLAID's shape: a stored zip inside a zip, reached at an offset."""
    inner = _zip_bytes({"2017/1.csv": b"1.0,2.0\n3.0,4.0\n", "2017/2.csv": b"5.0,6.0\n"})
    outer = _zip_bytes({"2017.zip": inner, "meta.json": b"[]"}, stored={"2017.zip", "meta.json"})
    url = host.put("plaid.zip", outer)

    nested = RemoteZip(url).nested("2017.zip")
    assert sorted(nested.entries) == ["2017/1.csv", "2017/2.csv"]
    assert b"".join(nested.stream("2017/1.csv")) == b"1.0,2.0\n3.0,4.0\n"


def test_a_deflated_member_cannot_be_read_in_place(host):
    url = host.put("z.zip", _zip_bytes({"inner.zip": b"x" * 4000}))
    with pytest.raises(FetchError, match="compressed"):
        RemoteZip(url).nested("inner.zip")


def test_a_host_that_ignores_ranges_is_an_error_not_a_download(host):
    url = host.put("z.zip", _zip_bytes({"a.txt": b"hi"}))
    host.ignore_ranges(True)
    with pytest.raises(FetchError, match="range requests"):
        _ = RemoteZip(url).entries


def test_an_encrypted_member_says_where_to_get_the_password(host):
    url = host.put("z.zip", _zip_bytes({"a.flac": b"hi"}))
    archive = RemoteZip(url)
    _ = archive.entries
    encrypted = archive.entries["a.flac"]
    archive._entries["a.flac"] = type(encrypted)(
        encrypted.name,
        encrypted.method,
        encrypted.compressed_size,
        encrypted.size,
        encrypted.header_offset,
        encrypted=True,
    )
    with pytest.raises(FetchError, match="encrypted"):
        list(archive.stream("a.flac"))


# --------------------------------------------------------------------------- #
# bounding a meter file in time
# --------------------------------------------------------------------------- #


def test_bounded_keeps_whole_lines_only():
    chunks = [b"100 5.0\n200 6.0\n30", b"0 7.0\n400 8.0\n"]
    assert b"".join(_bounded(iter(chunks), until=350)) == b"100 5.0\n200 6.0\n300 7.0\n"


def test_bounded_passes_everything_inside_the_window():
    chunks = [b"100 1\n", b"200 2\n"]
    assert b"".join(_bounded(iter(chunks), until=1e12)) == b"100 1\n200 2\n"


def test_bounded_emits_a_trailing_line_without_a_newline():
    assert b"".join(_bounded(iter([b"100 1\n200 2"]), until=1e12)) == b"100 1\n200 2"


def test_bounded_does_not_stop_on_an_unreadable_line():
    chunks = [b"garbage here\n100 1\n", b"900 9\n"]
    assert b"".join(_bounded(iter(chunks), until=500)) == b"garbage here\n100 1\n"


def test_bounded_extraction_truncates_the_written_file(host, tmp_path):
    readings = b"".join(f"{t} {t % 7}\n".encode() for t in range(1000, 9000, 6))
    url = host.put("ukdale.zip", _zip_bytes({"house_1/channel_1.dat": readings}))
    archive = RemoteZip(url)

    plan = Plan(
        "ukdale",
        (
            Artifact(
                url=url,
                relpath="low_freq/house_1/channel_1.dat",
                size_max=len(readings),
                member="house_1/channel_1.dat",
                archive_size=archive.size,
                stop_after_timestamp=2000,
            ),
        ),
    )
    fetch(plan, tmp_path, workers=1)
    written = (tmp_path / "low_freq/house_1/channel_1.dat").read_bytes()
    # Readings step by 6, so 1996 is the last one at or under the bound.
    assert written.endswith(b"1996 1\n")
    assert b"2002" not in written
    assert len(written) < len(readings)


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #


def _small_plan(host, tmp_name="a.txt", body=b"hello world"):
    url = host.put(tmp_name, body)
    import hashlib

    return Plan(
        "demo",
        (Artifact(url=url, relpath=tmp_name, size=len(body), md5=hashlib.md5(body).hexdigest()),),
    )


def test_fetch_writes_and_verifies(host, tmp_path):
    plan = _small_plan(host)
    report = fetch(plan, tmp_path, workers=1)
    assert (tmp_path / "a.txt").read_bytes() == b"hello world"
    assert report.fetched == ("a.txt",)
    assert report.nbytes == 11


def test_a_bad_checksum_is_refused(host, tmp_path):
    url = host.put("a.txt", b"hello world")
    plan = Plan("demo", (Artifact(url=url, relpath="a.txt", md5="0" * 32),))
    with pytest.raises(FetchError, match="MD5 mismatch"):
        fetch(plan, tmp_path, workers=1)
    assert not (tmp_path / "a.txt").exists()


def test_a_second_run_skips_what_is_already_there(host, tmp_path):
    plan = _small_plan(host)
    fetch(plan, tmp_path, workers=1)
    again = fetch(plan, tmp_path, workers=1)
    assert again.fetched == ()
    assert again.skipped == ("a.txt",)
    assert again.nbytes == 0


def test_force_refetches(host, tmp_path):
    plan = _small_plan(host)
    fetch(plan, tmp_path, workers=1)
    assert fetch(plan, tmp_path, workers=1, force=True).fetched == ("a.txt",)


def test_a_touched_file_is_fetched_again(host, tmp_path):
    plan = _small_plan(host)
    fetch(plan, tmp_path, workers=1)
    (tmp_path / "a.txt").write_bytes(b"tampered")
    assert fetch(plan, tmp_path, workers=1).fetched == ("a.txt",)


def test_a_dry_run_touches_nothing(host, tmp_path):
    plan = _small_plan(host)
    report = fetch(plan, tmp_path, dry_run=True)
    assert report.fetched == ("a.txt",)
    assert not (tmp_path / "a.txt").exists()


def test_a_plan_over_budget_is_refused_before_transferring(host, tmp_path):
    plan = _small_plan(host)
    with pytest.raises(FetchError, match="over the"):
        fetch(plan, tmp_path, max_bytes=5)
    assert not (tmp_path / "a.txt").exists()


def test_a_failure_is_reported_with_the_rest_still_fetched(host, tmp_path):
    good = host.put("good.txt", b"fine")
    plan = Plan(
        "demo",
        (
            Artifact(url=good, relpath="good.txt", size=4),
            Artifact(url=f"{host.base}/missing.txt", relpath="bad.txt"),
        ),
    )
    with pytest.raises(FetchError, match="1 of 2 artifacts failed"):
        fetch(plan, tmp_path, workers=1)
    assert (tmp_path / "good.txt").read_bytes() == b"fine"


def test_an_interrupted_download_resumes(host, tmp_path):
    body = b"0123456789" * 10
    url = host.put("a.bin", body)
    (tmp_path / "a.bin.part").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "a.bin.part").write_bytes(body[:40])

    plan = Plan("demo", (Artifact(url=url, relpath="a.bin", size=len(body)),))
    report = fetch(plan, tmp_path, workers=1)
    assert (tmp_path / "a.bin").read_bytes() == body
    assert report.nbytes == len(body)  # the file is whole, however it got here


def test_the_manifest_records_where_a_file_came_from(host, tmp_path):
    fetch(_small_plan(host), tmp_path, workers=1)
    manifest = json.loads((tmp_path / "nilmframe-cache.json").read_text())
    entry = manifest["entries"]["a.txt"]
    assert entry["url"].endswith("/a.txt")
    assert entry["size"] == 11


def test_a_narrower_cached_copy_does_not_satisfy_a_wider_request(tmp_path):
    ledger = _Ledger(tmp_path)
    artifact = Artifact(
        url="http://x/z.zip", relpath="c.dat", member="c.dat", stop_after_timestamp=1000
    )
    (tmp_path / "c.dat").write_bytes(b"x" * 10)
    ledger.record(artifact, 10)

    assert ledger.is_current(tmp_path, artifact)
    wider = Artifact(
        url="http://x/z.zip", relpath="c.dat", member="c.dat", stop_after_timestamp=2000
    )
    assert not ledger.is_current(tmp_path, wider)
    unbounded = Artifact(url="http://x/z.zip", relpath="c.dat", member="c.dat")
    assert not ledger.is_current(tmp_path, unbounded)


# --------------------------------------------------------------------------- #
# what the readers do with a fetched cache
# --------------------------------------------------------------------------- #


def test_ukdale_download_reaches_back_so_runs_enclose_the_waveforms(tmp_path, monkeypatch):
    """The meter window is widened, or every waveform lands in ``hf_only``.

    A waveform recorded on the hour starts before the first meter reading inside
    a window that begins on the same hour, so a run clipped to exactly the window
    does not contain it -- and a waveform outside every run loses the submeters
    that label it.
    """
    import nilmframe.sources as sources
    from nilmframe.readers import UKDALE

    root = tmp_path / "cache"
    (root / "low_freq" / "house_1").mkdir(parents=True)
    (root / "low_freq" / "house_1" / "labels.dat").write_text("1 aggregate\n")
    (root / "high_freq").mkdir()
    paths = {"dirpath": str(root / "low_freq"), "high_freq_root": str(root / "high_freq")}

    monkeypatch.setattr(UKDALE, "plan", classmethod(lambda cls, **kw: Plan("ukdale")))
    monkeypatch.setattr(sources, "materialize", lambda plan, cache, **kw: (paths, None))

    reader = UKDALE.download(root, houses=[1], time_range=(1_000.0, 2_000.0), progress=False)
    assert reader.time_range == (700.0, 2_000.0)  # one max_gap_s of slack


def test_credentials_never_reach_the_cache_manifest(tmp_path):
    """FTP artifacts carry an account in the URL; the manifest is not its home."""
    from nilmframe.sources.fetch import _Ledger

    ledger = _Ledger(tmp_path)
    artifact = Artifact(url="ftp://m1375836:m1375836@example.org/BLOND/a.hdf5", relpath="a.hdf5")
    (tmp_path / "a.hdf5").write_bytes(b"x")
    ledger.record(artifact, 1)
    ledger.flush()

    written = (tmp_path / "nilmframe-cache.json").read_text()
    assert "m1375836:m1375836" not in written
    assert "ftp://example.org/BLOND/a.hdf5" in written
    # ...and the sanitised record still recognises the same artifact next run.
    assert ledger.is_current(tmp_path, artifact)


def test_a_zip_index_is_reused_from_disk(host, tmp_path):
    """A 770k-member directory is 85 MB of footer; reading it twice is the bug."""
    url = host.put("big.zip", _zip_bytes({"a.txt": b"hello", "b/c.txt": b"world"}))
    first = RemoteZip(url, index_cache=tmp_path)
    assert sorted(first.entries) == ["a.txt", "b/c.txt"]
    assert list(tmp_path.glob("zipindex-*.json"))

    # Take the server away: a second reader must answer from the cache alone.
    host.ignore_ranges(True)
    second = RemoteZip(url, index_cache=tmp_path)
    assert second.entries["b/c.txt"].size == 5
    assert second.entries["a.txt"].method == first.entries["a.txt"].method


def test_blond_filenames_carry_the_clock():
    from nilmframe.readers.blond import parse_blond_name

    info = parse_blond_name("clear-2016-09-30T11-05-36.873488T+0200-0000000.hdf5")
    assert info["unit"] == "clear"
    # 11:05:36 at +0200 is 09:05:36 UTC.
    assert info["t0"] == pytest.approx(1475226336.873488)
    medal = parse_blond_name("medal-15-2017-04-30T23-59-59.000001T+0200-0016925.hdf5")
    assert medal["unit"] == "medal-15"  # the unit name has a dash of its own
    assert medal["sequence"] == 16925
    assert parse_blond_name("summary-2016-09-30-medal-1.hdf5") is None


def test_blond_labels_follow_the_log_as_it_stood(tmp_path):
    """Sockets were re-used over seven months; the newest prior entry wins."""
    from nilmframe.readers.blond import BLOND

    log = {
        "MEDAL-1": {
            "circuit_id": "L1",
            "entries": [
                {"timestamp": "2016-09-28T17-00-00", "socket_1": {"class_name": "Fan"}},
                {"timestamp": "2017-01-15T09-00-00", "socket_1": {"class_name": "Monitor"}},
            ],
        }
    }
    (tmp_path / "2016-09-30").mkdir()
    reader = BLOND(tmp_path, appliance_log=log)
    early = reader._config_at("medal-1", datetime(2016, 10, 1))
    late = reader._config_at("medal-1", datetime(2017, 2, 1))
    assert early["socket_1"]["class_name"] == "Fan"
    assert late["socket_1"]["class_name"] == "Monitor"
    # Before the first entry there is nothing in force; fall back, do not crash.
    assert reader._config_at("medal-1", datetime(2015, 1, 1))["socket_1"]["class_name"] == "Fan"


def test_blond_days_widen_around_a_time_range():
    from nilmframe.sources.blond import _days_in

    days = _days_in((1475226336.0, 1475226336.0))
    assert days == {"2016-09-29", "2016-09-30", "2016-10-01"}


def test_whited_plan_covers_only_the_single_appliance_corpus():
    """The archive carries three subtrees that are not the corpus.

    ``Experiments/`` (including simultaneous two-appliance runs), ``notUsed/`` and
    ``MIXED/`` have filenames shaped like corpus recordings but are not, and the
    reader's non-recursive glob never sees them. Fetching them was 527 MiB of
    files that would then be ignored.
    """
    from nilmframe.sources import WHITEDSource
    from nilmframe.sources._zip import ZipEntry

    names = [
        "DATEN/Kettle_1800W_r1_MK2_20151009173102.flac",
        "DATEN/Fan_ChingHai35W_r6_MK1_20151229103607.flac",
        "DATEN/Experiments/2Appl/Fan_ChingHai35W_r6_MK1_20151229103625.flac",
        "DATEN/notUsed/Fan_ChingHai35W_r6_MK1_20151229103652.flac",
        "DATEN/MIXED/Fan_ChingHai35W_r6_MK1_20151229103713.flac",
    ]

    class _Archive:
        size = 1_000
        entries = {n: ZipEntry(n, 8, 100, 200, 0) for n in names}

    source = WHITEDSource()
    source._archive = _Archive()
    plan = source.plan()

    assert sorted(a.relpath for a in plan.artifacts) == [
        "flac/Fan_ChingHai35W_r6_MK1_20151229103607.flac",
        "flac/Kettle_1800W_r1_MK2_20151009173102.flac",
    ]
    assert any("not part of the single-appliance corpus" in note for note in plan.notes)


def test_plaid_reads_the_shapes_it_is_published_in():
    """The release ships a list of records split across two files, not a mapping."""
    from nilmframe.readers.plaid import PLAID

    meta = {"header": {"sampling_frequency": "30000Hz"}, "appliance": {"type": "Kettle"}}
    mapping = PLAID._as_records({"7": meta})
    listed = PLAID._as_records([{"id": 7, "meta": meta}])
    assert mapping == listed == {"7": meta}

    with pytest.raises(TypeError, match="paths or"):
        PLAID._as_records([{"no": "id"}])
    with pytest.raises(TypeError, match="mapping, a list"):
        PLAID._as_records(42)


def test_plaid_merges_several_metadata_files(tmp_path):
    from nilmframe.readers.plaid import PLAID

    header = {"header": {"sampling_frequency": "30000Hz"}}
    for name, ids in (("a.json", ["1"]), ("b.json", ["2"])):
        (tmp_path / name).write_text(
            json.dumps(
                [{"id": i, "meta": {**header, "appliance": {"type": "Kettle"}}} for i in ids]
            )
        )
    for i in ("1", "2"):
        (tmp_path / f"{i}.csv").write_text("1.0,2.0\n3.0,4.0\n")

    reader = PLAID(tmp_path, [tmp_path / "a.json", tmp_path / "b.json"])
    assert [rec.session for rec in reader] == ["1", "2"]


def test_a_whole_cached_copy_satisfies_a_bounded_request(tmp_path):
    ledger = _Ledger(tmp_path)
    whole = Artifact(url="http://x/z.zip", relpath="c.dat", member="c.dat", size=10)
    (tmp_path / "c.dat").write_bytes(b"x" * 10)
    ledger.record(whole, 10)
    bounded = Artifact(
        url="http://x/z.zip", relpath="c.dat", member="c.dat", stop_after_timestamp=2000
    )
    assert ledger.is_current(tmp_path, bounded)
