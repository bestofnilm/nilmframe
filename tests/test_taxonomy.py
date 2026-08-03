"""Appliance labels: reconciling the vocabularies before a merge."""

from __future__ import annotations

import numpy as np
import pytest

from nilmframe.store import ChannelKind, Recording, Store, StoreWriter
from nilmframe.store.merge import merge_stores
from nilmframe.taxonomy import (
    APPLIANCE_ALIASES,
    Taxonomy,
    default_taxonomy,
    normalise,
)

# --------------------------------------------------------------------------- #
# Normalisation -- the mechanical half
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Fridge", "fridge"),
        ("Washing Machine", "washing_machine"),
        ("washing_machine", "washing_machine"),
        ("Fridge-Freezer", "fridge_freezer"),
        ("Fridge Freezer", "fridge_freezer"),
        ("Fridge-Freezer(2)", "fridge_freezer"),
        ("Freezer( 1 )", "freezer"),
        ("TV/Satellite", "tv_satellite"),
        ("  Kettle  ", "kettle"),
        ("???", ""),
        ("", ""),
    ],
)
def test_normalise_collapses_the_mechanical_variation(label, expected):
    assert normalise(label) == expected


def test_normalise_keeps_parenthesised_words():
    """A word in brackets is as likely the appliance as the room it sits in.

    ``Magimix(Blender)`` names the appliance and ``Freezer(garage)`` names a
    location. Dropping either silently would merge two different loads, so the
    normaliser keeps both and the alias table decides.
    """
    assert normalise("Magimix(Blender)") == "magimix_blender"
    assert normalise("Freezer(garage)") == "freezer_garage"


def test_instance_number_is_not_identity():
    """Two freezers in one house are one appliance type; instance_id carries which."""
    assert normalise("Freezer(1)") == normalise("Freezer(2)") == "freezer"


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def test_canonical_name_is_its_own_alias():
    taxonomy = Taxonomy({"fridge": ["refrigerator"]})
    assert taxonomy.resolve("fridge") == "fridge"
    assert taxonomy.resolve("Fridge") == "fridge"


def test_unknown_label_returns_none_and_does_not_guess():
    assert Taxonomy({"fridge": []}).resolve("vivarium") is None


def test_strict_refuses_rather_than_passing_through():
    taxonomy = Taxonomy({"fridge": []}, strict=True)
    with pytest.raises(KeyError, match="no canonical name for 'vivarium'"):
        taxonomy.resolve("vivarium")


def test_none_passes_through_because_mains_has_no_appliance():
    assert Taxonomy({"fridge": []}, strict=True).resolve(None) is None


def test_contradictory_alias_is_rejected_at_construction():
    """Silently resolving by dict order is the one thing worse than an error."""
    with pytest.raises(ValueError, match="maps to both"):
        Taxonomy({"fridge": ["cooler"], "freezer": ["cooler"]})


def test_per_dataset_override_beats_the_global_table():
    """No global table can say that one corpus's `washer` is a dishwasher."""
    taxonomy = Taxonomy(
        {"washing_machine": ["washer"]},
        per_dataset={"odd_corpus": {"washer": "dishwasher"}},
    )
    assert taxonomy.resolve("washer", "refit") == "washing_machine"
    assert taxonomy.resolve("washer", "odd_corpus") == "dishwasher"
    assert taxonomy.resolve("washer") == "washing_machine"


def test_taxonomy_is_callable():
    taxonomy = Taxonomy({"fridge": ["refrigerator"]})
    assert taxonomy("refrigerator") == "fridge"


def test_with_aliases_extends_without_mutating():
    base = default_taxonomy()
    extended = base.with_aliases(kettle=["K Mix"])
    assert extended.resolve("K Mix") == "kettle"
    assert base.resolve("K Mix") is None, "the original must not be touched"


# --------------------------------------------------------------------------- #
# The shipped table, against labels that really exist
# --------------------------------------------------------------------------- #


def test_shipped_table_handles_refits_published_typos():
    """`Dishwaser` and `Firdge` are in the released dataset and will not be fixed."""
    taxonomy = default_taxonomy()
    assert taxonomy.resolve("Dishwaser") == "dishwasher"
    assert taxonomy.resolve("Firdge") == "fridge"


def test_shipped_table_covers_most_of_refit_and_admits_the_rest():
    from nilmframe.readers.refit import APPLIANCES

    taxonomy = default_taxonomy()
    labels = sorted({name for house in APPLIANCES.values() for name in house.values()})
    unmapped = [name for name in labels if taxonomy.resolve(name) is None]

    # The ones left are genuinely ambiguous, not merely unlisted: a Kenwood kMix
    # is a kettle in one product line and a stand mixer in another, and `???` is
    # what the dataset itself says.
    assert unmapped == ["???", "K Mix", "Magimix(Blender)", "Pond Pump", "Vivarium"]


def test_every_alias_in_the_shipped_table_resolves_to_its_own_canonical():
    taxonomy = default_taxonomy()
    for canonical, aliases in APPLIANCE_ALIASES.items():
        for alias in (canonical, *aliases):
            assert taxonomy.resolve(alias) == canonical, alias


def test_every_shipped_canonical_name_has_a_category():
    taxonomy = default_taxonomy()
    without = [name for name in taxonomy.canonical if taxonomy.category(name) == "unknown"]
    assert without == []


# --------------------------------------------------------------------------- #
# Inspecting before merging
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_vocabularies(tmp_path):
    """Two stores that call the same three appliances different things."""

    def build(where, dataset, labels):
        with StoreWriter(where) as writer:
            for label in labels:
                writer.add(
                    Recording(
                        dataset=dataset,
                        house="h",
                        session="s",
                        kind=ChannelKind.SUBMETER,
                        appliance=label,
                        signals={"p": np.full(200, 50.0, np.float32)},
                        fs=100.0,
                    )
                )
        return Store(where)

    a = build(tmp_path / "a", "alpha", ["refrigerator", "Washing Machine", "Vivarium"])
    b = build(tmp_path / "b", "beta", ["Firdge", "washer", "Dishwaser"])
    return a, b


def test_report_is_a_dry_run(two_vocabularies):
    a, b = two_vocabularies
    report = default_taxonomy().report(a, b)

    assert list(report.columns) == [
        "dataset",
        "label",
        "normalised",
        "canonical",
        "category",
        "mapped",
    ]
    assert len(report) == 6
    # Unmapped first: those are the rows that need a decision.
    assert not bool(report.iloc[0]["mapped"])
    assert report.iloc[0]["label"] == "Vivarium"

    mapping = dict(zip(report["label"], report["canonical"], strict=True))
    assert mapping["refrigerator"] == "fridge"
    assert mapping["Firdge"] == "fridge"
    assert mapping["washer"] == "washing_machine"
    assert mapping["Vivarium"] == "Vivarium", "unmapped labels are reported unchanged"


def test_unmapped_lists_what_nothing_claims(two_vocabularies):
    a, b = two_vocabularies
    assert default_taxonomy().unmapped(a, b) == ["Vivarium"]


def test_unmapped_does_not_raise_under_strict(two_vocabularies):
    """Asking what is missing must work on the taxonomy that refuses to guess."""
    a, b = two_vocabularies
    taxonomy = default_taxonomy(strict=True)
    assert taxonomy.unmapped(a, b) == ["Vivarium"]
    assert taxonomy.strict is True, "the flag must be restored"


def test_as_dict_only_contains_labels_that_change(two_vocabularies):
    a, b = two_vocabularies
    mapping = default_taxonomy().as_dict(a, b)
    assert mapping == {
        "refrigerator": "fridge",
        "Firdge": "fridge",
        "Washing Machine": "washing_machine",
        "washer": "washing_machine",
        "Dishwaser": "dishwasher",
    }


# --------------------------------------------------------------------------- #
# Through the merge
# --------------------------------------------------------------------------- #


def test_merging_under_a_taxonomy_collapses_the_label_space(two_vocabularies, tmp_path):
    a, b = two_vocabularies

    naive = merge_stores([a, b], tmp_path / "naive")
    assert len(naive.appliances) == 6, "six spellings, six classes"

    merged = merge_stores([a, b], tmp_path / "merged", taxonomy=default_taxonomy())
    assert merged.appliances == ["Vivarium", "dishwasher", "fridge", "washing_machine"]
    assert sorted(merged.channels["appliance"].unique()) == [
        "Vivarium",
        "dishwasher",
        "fridge",
        "washing_machine",
    ]


def test_unknown_labels_survive_the_merge(two_vocabularies, tmp_path):
    """Leaving a label alone is recoverable; dropping the channel is not."""
    a, b = two_vocabularies
    merged = merge_stores([a, b], tmp_path / "m", taxonomy=default_taxonomy())
    assert "Vivarium" in merged.appliances


def test_taxonomy_fills_the_category_column(two_vocabularies, tmp_path):
    a, b = two_vocabularies
    merged = merge_stores([a, b], tmp_path / "m", taxonomy=default_taxonomy())
    categories = dict(
        zip(merged.appliance_table["appliance"], merged.appliance_table["category"], strict=True)
    )
    assert categories["fridge"] == "cold"
    assert categories["washing_machine"] == "wet"
    assert categories["Vivarium"] == "unknown"


def test_rename_overrides_the_taxonomy(two_vocabularies, tmp_path):
    """`rename` is the last word, or it would not be an override."""
    a, b = two_vocabularies
    merged = merge_stores(
        [a, b],
        tmp_path / "m",
        taxonomy=default_taxonomy(),
        rename={"fridge": "cold_appliance"},
    )
    assert "cold_appliance" in merged.appliances
    assert "fridge" not in merged.appliances


def test_a_plain_dict_is_accepted_as_a_taxonomy(two_vocabularies, tmp_path):
    a, b = two_vocabularies
    merged = merge_stores([a, b], tmp_path / "m", taxonomy={"fridge": ["refrigerator", "Firdge"]})
    assert "fridge" in merged.appliances
    assert "refrigerator" not in merged.appliances


def test_per_dataset_override_reaches_the_channels(tmp_path):
    """The overrides are useless if the merge cannot tell the corpora apart."""

    def build(where, dataset):
        with StoreWriter(where) as writer:
            writer.add(
                Recording(
                    dataset=dataset,
                    house="h",
                    session="s",
                    kind=ChannelKind.SUBMETER,
                    appliance="washer",
                    signals={"p": np.full(200, 50.0, np.float32)},
                    fs=100.0,
                )
            )
        return Store(where)

    a, b = build(tmp_path / "a", "alpha"), build(tmp_path / "b", "beta")
    taxonomy = Taxonomy(
        {"washing_machine": ["washer"]}, per_dataset={"beta": {"washer": "dishwasher"}}
    )

    merged = merge_stores([a, b], tmp_path / "m", taxonomy=taxonomy)
    assert sorted(merged.appliances) == ["dishwasher", "washing_machine"]


def test_taxonomy_applies_to_activations(plaid_store, tmp_path):
    merged = merge_stores([plaid_store], tmp_path / "m", taxonomy={"water_heater": ["kettle"]})
    assert "water_heater" in merged.appliances
    assert set(merged.activations["appliance"]) <= set(merged.appliances)


def test_the_resolved_map_is_recorded_in_the_manifest(two_vocabularies, tmp_path):
    """A merged store has to say how its labels were arrived at, years later."""
    a, b = two_vocabularies
    merged = merge_stores([a, b], tmp_path / "m", taxonomy=default_taxonomy())

    recorded = merged.manifest["merge_rules"]["taxonomy"]
    assert recorded["Firdge"] == "fridge"
    assert "Vivarium" not in recorded, "only labels that changed"

    import json

    json.dumps(recorded), "the manifest must stay JSON"


def test_merging_without_a_taxonomy_is_unchanged(two_vocabularies, tmp_path):
    a, b = two_vocabularies
    merged = merge_stores([a, b], tmp_path / "m")
    assert merged.manifest["merge_rules"]["taxonomy"] == {}
    assert len(merged.appliances) == 6
