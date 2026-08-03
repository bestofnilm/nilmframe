"""Appliance labels: the same appliance under a dozen names.

Every corpus names its loads in its own way, and none of them is wrong. UK-DALE
writes ``fridge``, another writes ``refrigerator``; REFIT writes ``Washing
Machine``, ``Washing Machine(1)`` and ``Washing Machine(2)`` for what is one
appliance type in three houses. Merge two stores without reconciling that and the
result has twice the vocabulary and half the data behind each label, which is
worse than either input: a model trained on it learns that ``fridge`` and
``refrigerator`` are different things.

REFIT alone, from :data:`nilmframe.readers.refit.APPLIANCES`, is the argument for
this module. It has 48 distinct labels for maybe twenty appliance types, and the
variation is of five different kinds:

``Dishwaser``, ``Firdge``
    typos, in the published dataset, which will never be fixed.
``Fridge Freezer``, ``Fridge-Freezer``
    the same words under different separators.
``Washing Machine(1)``, ``Freezer(2)``
    an instance number, because that house has two.
``Television Site``, ``Computer Site``
    a circuit feeding an appliance rather than the appliance.
``PGM Computer``, ``MJY Computer``
    a household member's initials.

No amount of string normalisation resolves all five. Case, separators and instance
numbers are mechanical, and :func:`normalise` handles them. The rest is knowledge
about the corpus, and knowledge has to be written down -- which is what a
:class:`Taxonomy` is.

Two things follow from writing it down rather than inlining a dict at the merge:

**A mapping can be inspected before it is applied.** :meth:`Taxonomy.report` shows
what every label in a set of stores would become, and :meth:`Taxonomy.unmapped`
lists the ones nothing claims. A merge is expensive and writes a new store; being
able to see that ``K Mix`` went unrecognised *first* is the difference between a
taxonomy you trust and a dict you hope about.

**Silence stops being the failure mode.** A plain ``rename={"fridge": "fridge"}``
with a typo in the key does nothing and says nothing. A taxonomy in
``strict=True`` refuses to resolve a label it does not know.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import cycle, and only needed for typing
    import pandas as pd

    from nilmframe.store import Store

__all__ = ["Taxonomy", "default_taxonomy", "normalise"]

#: A parenthesised integer is an instance number -- REFIT's ``Freezer(1)`` and
#: ``Freezer(2)`` are one type in one house. Instance identity is already carried
#: by the schema's ``instance_id``, so it does not belong in the label too.
_INSTANCE = re.compile(r"\(\s*\d+\s*\)")

#: What separates words. Brackets are in here rather than simply deleted: a
#: parenthesised *word* is as likely to be the appliance (``Magimix(Blender)``) as
#: a location (``Freezer(garage)``), so it is kept as part of the key and the
#: alias table decides. Deleting it would silently merge two different loads.
_SEPARATORS = re.compile(r"[\s\-/,.()\[\]]+")

_TRIM = re.compile(r"[^a-z0-9_]+")


def normalise(label: str) -> str:
    """Reduce a label to the form aliases are matched on.

    Case-folds, drops instance numbers, and unifies separators to ``_``. This is
    the mechanical half of the problem only -- it makes ``Fridge-Freezer``,
    ``fridge freezer`` and ``Fridge_Freezer(2)`` one key, and does nothing at all
    about ``Firdge``.

    Args:
        label: a label as some corpus wrote it.

    Returns:
        The normalised form. Empty string for an empty or punctuation-only label.

    Example:
        >>> from nilmframe.taxonomy import normalise
        >>> normalise('Fridge-Freezer(2)')
        'fridge_freezer'
        >>> normalise('Washing Machine'), normalise('washing_machine')
        ('washing_machine', 'washing_machine')
        >>> normalise('???')
        ''
    """
    text = _INSTANCE.sub(" ", str(label).casefold())
    text = _SEPARATORS.sub("_", text.strip())
    return _TRIM.sub("", text).strip("_")


class Taxonomy:
    """A mapping from the labels corpora use to the ones you want to model.

    Written canonical-name-first, because that is the direction a person thinks
    in -- *these are all the fridge* -- and inverted internally, because that is
    the direction a lookup needs. Aliases are matched on their :func:`normalise`
    form, so a table does not need an entry per separator and per instance number.

    Args:
        aliases: canonical name to the labels that mean it. The canonical name is
            always an alias of itself and does not need repeating.
        categories: canonical name to a coarse group. Fills the store's
            ``category`` column, which is otherwise ``unknown``.
        per_dataset: dataset name to overrides applied before ``aliases``. For
            labels that are genuinely ambiguous between corpora rather than merely
            spelled differently -- if one corpus's ``washer`` is a washing machine
            and another's is a dishwasher, no global table can express that.
        strict: raise on a label nothing claims, instead of returning ``None``.

    Example:
        >>> from nilmframe.taxonomy import Taxonomy
        >>> taxonomy = Taxonomy({'fridge': ['refrigerator', 'Firdge'],
        ...                      'washing_machine': ['Washing Machine', 'washer']})
        >>> taxonomy.resolve('Refrigerator')
        'fridge'
        >>> taxonomy.resolve('Washing Machine(2)')
        'washing_machine'
        >>> taxonomy.resolve('vivarium') is None
        True
    """

    def __init__(
        self,
        aliases: Mapping[str, Sequence[str]] | None = None,
        *,
        categories: Mapping[str, str] | None = None,
        per_dataset: Mapping[str, Mapping[str, str]] | None = None,
        strict: bool = False,
    ) -> None:
        self._lookup: dict[str, str] = {}
        self._aliases: dict[str, list[str]] = {}
        for canonical, names in (aliases or {}).items():
            self._aliases[canonical] = list(names)
            for name in (canonical, *names):
                key = normalise(name)
                if not key:
                    continue
                previous = self._lookup.get(key)
                if previous is not None and previous != canonical:
                    raise ValueError(
                        f"alias {name!r} maps to both {previous!r} and {canonical!r}; "
                        "an alias table that contradicts itself resolves by dict order"
                    )
                self._lookup[key] = canonical

        self._categories = dict(categories or {})
        self._per_dataset = {
            normalise(dataset): {normalise(k): v for k, v in table.items()}
            for dataset, table in (per_dataset or {}).items()
        }
        self.strict = strict

    # -- lookup ------------------------------------------------------------- #

    def resolve(self, label: str | None, dataset: str | None = None) -> str | None:
        """The canonical name for a label, or ``None`` if nothing claims it.

        Args:
            label: the label as the corpus wrote it. ``None`` passes through --
                a mains channel has no appliance and that is not a failure.
            dataset: which corpus it came from, consulted for ``per_dataset``
                overrides before the global table.

        Returns:
            The canonical name, or ``None`` when unknown and not ``strict``.

        Raises:
            KeyError: when unknown and ``strict`` is set.

        Example:
            >>> from nilmframe.taxonomy import default_taxonomy
            >>> default_taxonomy().resolve('Dishwaser')
            'dishwasher'
        """
        if label is None:
            return None
        key = normalise(label)
        if not key:
            return self._miss(label)

        if dataset is not None:
            override = self._per_dataset.get(normalise(dataset), {}).get(key)
            if override is not None:
                return override

        found = self._lookup.get(key)
        return found if found is not None else self._miss(label)

    def _miss(self, label: str) -> None:
        if self.strict:
            raise KeyError(
                f"no canonical name for {label!r}. Add it to the taxonomy, or pass "
                "strict=False to leave unknown labels as they are."
            )
        return None

    def __call__(self, label: str | None, dataset: str | None = None) -> str | None:
        """:meth:`resolve`, so a taxonomy can be used where a function is expected."""
        return self.resolve(label, dataset)

    def category(self, canonical: str) -> str:
        """The coarse group a canonical name belongs to, or ``'unknown'``.

        Example:
            >>> from nilmframe.taxonomy import default_taxonomy
            >>> default_taxonomy().category('dishwasher')
            'wet'
        """
        return self._categories.get(canonical, "unknown")

    @property
    def canonical(self) -> list[str]:
        """Every canonical name, sorted."""
        return sorted(self._aliases)

    # -- inspection --------------------------------------------------------- #

    def as_dict(self, *stores: Store) -> dict[str, str]:
        """A flat ``{label: canonical}`` map for the labels these stores contain.

        Only labels that actually resolve appear, so applying it leaves anything
        unrecognised untouched. This is what gets written into a merged store's
        manifest: a dict survives being read back in five years, an object of this
        class does not.

        With no stores it returns the whole table instead, keyed by
        :func:`normalise`\\ d alias rather than by any corpus's spelling.

        Example:
            >>> from nilmframe.taxonomy import default_taxonomy
            >>> mapping = default_taxonomy().as_dict()
            >>> mapping['dishwaser'], mapping['refrigerator']
            ('dishwasher', 'fridge')
        """
        if not stores:
            return dict(self._lookup)
        out: dict[str, str] = {}
        for dataset, label in self._labels(stores):
            canonical = self.resolve(label, dataset)
            if canonical is not None and canonical != label:
                out[label] = canonical
        return out

    def report(self, *stores: Store) -> pd.DataFrame:
        """What every label in these stores would become. A dry run.

        Returns:
            A frame of ``dataset``, ``label``, ``normalised``, ``canonical``,
            ``category`` and ``mapped``, sorted with the unmapped rows first --
            those are the ones that need a decision.

        Example:
            >>> from nilmframe.taxonomy import default_taxonomy
            >>> report = default_taxonomy().report(store)
            >>> list(report.columns)
            ['dataset', 'label', 'normalised', 'canonical', 'category', 'mapped']
            >>> bool(report['mapped'].all())
            True
        """
        import pandas as pd

        rows = []
        for dataset, label in self._labels(stores):
            canonical = Taxonomy.resolve(self, label, dataset)  # never raise here
            rows.append(
                {
                    "dataset": dataset,
                    "label": label,
                    "normalised": normalise(label),
                    "canonical": canonical if canonical is not None else label,
                    "category": self.category(canonical) if canonical else "unknown",
                    "mapped": canonical is not None,
                }
            )
        frame = pd.DataFrame(
            rows, columns=["dataset", "label", "normalised", "canonical", "category", "mapped"]
        )
        return frame.sort_values(["mapped", "dataset", "label"]).reset_index(drop=True)

    def unmapped(self, *stores: Store) -> list[str]:
        """Labels in these stores that nothing in the table claims.

        The list to work through before trusting a merge.

        Example:
            >>> from nilmframe.taxonomy import Taxonomy
            >>> Taxonomy({'fridge': ['refrigerator']}).unmapped(store)
            ['kettle', 'laptop', 'microwave']
        """
        strict, self.strict = self.strict, False
        try:
            missing = {
                label
                for dataset, label in self._labels(stores)
                if self.resolve(label, dataset) is None
            }
        finally:
            self.strict = strict
        return sorted(missing)

    def with_aliases(self, **aliases: Sequence[str]) -> Taxonomy:
        """A copy with more aliases. The shipped table is a starting point.

        Example:
            >>> from nilmframe.taxonomy import default_taxonomy
            >>> mine = default_taxonomy().with_aliases(kettle=['K Mix'])
            >>> mine.resolve('K Mix')
            'kettle'
        """
        merged = {name: list(names) for name, names in self._aliases.items()}
        for canonical, names in aliases.items():
            merged.setdefault(canonical, [])
            merged[canonical].extend(names)
        return Taxonomy(
            merged,
            categories=self._categories,
            per_dataset={k: dict(v) for k, v in self._per_dataset.items()},
            strict=self.strict,
        )

    def __repr__(self) -> str:
        return (
            f"Taxonomy({len(self._aliases)} canonical, {len(self._lookup)} aliases"
            f"{', strict' if self.strict else ''})"
        )

    # -- internals ---------------------------------------------------------- #

    @staticmethod
    def _labels(stores: Iterable[Store]) -> list[tuple[str, str]]:
        """``(dataset, label)`` for every appliance in every store, deduplicated."""
        seen: dict[tuple[str, str], None] = {}
        for store in stores:
            frame = store.channels
            for _, row in frame.iterrows():
                label = row.get("appliance")
                if label is None or (isinstance(label, float) and label != label):
                    continue
                seen.setdefault((str(row.get("dataset", "")), str(label)), None)
        return sorted(seen)


# --------------------------------------------------------------------------- #
# The shipped table.

#: Canonical name to the labels that mean it, for the corpora this package reads.
#:
#: Every REFIT entry here is one of the 48 strings in
#: :data:`nilmframe.readers.refit.APPLIANCES`, and every UCI entry one of the three
#: in :data:`nilmframe.readers.uci.SUBMETERS`. The rest are the spellings the other
#: corpora use for the same loads. This is a starting point and not an authority:
#: extend it with :meth:`Taxonomy.with_aliases` rather than expecting it to be
#: complete for your merge.
#:
#: Deliberately absent: ``K Mix`` (a Kenwood kMix, which is a kettle in one product
#: line and a stand mixer in another), ``Magimix(Blender)``, ``Vivarium``,
#: ``Pond Pump`` and REFIT's ``???``. Guessing at those would put two different
#: loads under one label, which is the failure this module exists to prevent, so
#: they are left for :meth:`Taxonomy.unmapped` to surface.
APPLIANCE_ALIASES: dict[str, list[str]] = {
    # Cold.
    "fridge": ["refrigerator", "Firdge", "Fridge(garage)", "fridge_garage"],
    "freezer": ["Chest Freezer", "Freezer(garage)", "freezer_garage"],
    "fridge_freezer": ["Fridge Freezer", "Fridge-Freezer", "fridge freezer"],
    # Wet.
    "washing_machine": ["Washing Machine", "washer", "washingmachine"],
    "tumble_dryer": ["Tumble Dryer", "dryer", "tumbledryer", "clothes dryer"],
    "washer_dryer": ["Washer Dryer", "Washer Dryer(garage)", "washer_dryer_garage"],
    "dishwasher": ["Dishwaser", "dish washer"],
    # Kitchen.
    "kettle": ["electric kettle"],
    "microwave": ["microwave oven"],
    "toaster": [],
    "oven": ["electric oven", "cooker"],
    "hob": ["cooktop", "stove", "hotplate"],
    "bread_maker": ["Bread-maker", "breadmaker", "bread maker"],
    "food_mixer": ["Food Mixer", "mixer"],
    "coffee_machine": ["coffee maker", "coffeemaker", "espresso machine"],
    "blender": [],
    "kitchen_outlets": ["kitchen"],
    # Electronics.
    "television": [
        "Television Site",
        "TV",
        "TV/Satellite",
        "TV Site(Bedroom)",
        "tv_site_bedroom",
        "telly",
    ],
    "computer": [
        "Computer Site",
        "Desktop Computer",
        "PC",
        "MJY Computer",
        "PGM Computer",
        "laptop computer",
    ],
    "laptop": [],
    "monitor": ["screen", "display"],
    "router": ["Network Site", "modem", "network"],
    "games_console": ["Games Console", "console"],
    "hifi": ["Hi-Fi", "stereo", "audio system"],
    "printer": [],
    # Heating, cooling, air.
    "electric_heater": ["Electric Heater", "heater", "space heater", "electric heating"],
    "water_heater": ["boiler", "immersion heater", "water_heater_air_conditioner"],
    "air_conditioner": ["air conditioning", "aircon", "AC"],
    "fan": ["Overhead Fan", "ceiling fan"],
    "dehumidifier": [],
    "vacuum_cleaner": ["vacuum", "hoover"],
    # Everything else.
    "lighting": ["lights", "light", "lamp"],
    "laundry_room": [],
    "solar": ["pv", "photovoltaic"],
}

#: Canonical name to a coarse group, for the store's ``category`` column.
APPLIANCE_CATEGORIES: dict[str, str] = {
    "fridge": "cold",
    "freezer": "cold",
    "fridge_freezer": "cold",
    "washing_machine": "wet",
    "tumble_dryer": "wet",
    "washer_dryer": "wet",
    "dishwasher": "wet",
    "kettle": "kitchen",
    "microwave": "kitchen",
    "toaster": "kitchen",
    "oven": "kitchen",
    "hob": "kitchen",
    "bread_maker": "kitchen",
    "food_mixer": "kitchen",
    "coffee_machine": "kitchen",
    "blender": "kitchen",
    "kitchen_outlets": "kitchen",
    "television": "electronics",
    "computer": "electronics",
    "laptop": "electronics",
    "monitor": "electronics",
    "router": "electronics",
    "games_console": "electronics",
    "hifi": "electronics",
    "printer": "electronics",
    "electric_heater": "hvac",
    "water_heater": "hvac",
    "air_conditioner": "hvac",
    "fan": "hvac",
    "dehumidifier": "hvac",
    "vacuum_cleaner": "other",
    "lighting": "lighting",
    "laundry_room": "wet",
    "solar": "generation",
}


def default_taxonomy(strict: bool = False) -> Taxonomy:
    """The shipped alias table, as a :class:`Taxonomy`.

    Args:
        strict: raise on a label the table does not know, rather than leaving it.

    Example:
        >>> from nilmframe.taxonomy import default_taxonomy
        >>> taxonomy = default_taxonomy()
        >>> taxonomy.resolve('Fridge-Freezer(2)')
        'fridge_freezer'
        >>> taxonomy.resolve('Television Site')
        'television'
        >>> taxonomy.category('television')
        'electronics'
    """
    return Taxonomy(
        APPLIANCE_ALIASES,
        categories=APPLIANCE_CATEGORIES,
        strict=strict,
    )
