import math
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import plotly.graph_objects as go

from connectome_interpreter.external_map import (
    hex_heatmap,
    load_dataset,
    plot_mollweide_projection,
)

EYEMAP_ROWS = [
    # hex1, hex2 = q, p -- solved from hex_heatmap()'s background_hex build,
    # x = hex2 - hex1, y = hex2 + hex1, matching x,y = p - q, p + q below.
    {"p": 0, "q": 0, "x": 0.0, "y": 0.0, "z": 1.0, "hex1": 0, "hex2": 0},
    {"p": 1, "q": 0, "x": 0.1, "y": 0.0, "z": 0.99, "hex1": 0, "hex2": 1},
    {"p": 0, "q": 1, "x": 0.0, "y": 0.1, "z": 0.99, "hex1": 1, "hex2": 0},
    {"p": -1, "q": 0, "x": -0.1, "y": 0.0, "z": 0.99, "hex1": 0, "hex2": -1},
    {"p": 0, "q": -1, "x": 0.0, "y": -0.1, "z": 0.99, "hex1": -1, "hex2": 0},
]

# Half-gap between the eyes in the two-eye fixture. Any even value above the
# single-eye x half-width keeps the two blocks clear of each other; the real
# male-CNS eyemaps use 20.
TWO_EYE_X_OFFSET = 4

MM_PER_PX = 25.4 / 72  # hex_heatmap sizes in mm and lays out in px at dpi=72
CBAR_MM = 54  # room hex_heatmap leaves for the colorbar


def binocular_rows(side: str, offset: int = TWO_EYE_X_OFFSET) -> list[dict]:
    """
    Build one hemisphere's eyemap fixture, with ``hex1b``/``hex2b`` appended.

    Mirrors what the eyemap archive's ``download_eyemaps.add_binocular_hex()``
    writes: ``hex1``/``hex2`` are left exactly as they are, and the binocular
    pair encodes ``x = hex2 - hex1`` shifted to ``x + offset`` for the right eye
    and ``-(x + offset)`` for the left, with ``y = hex2 + hex1`` untouched.
    Mirroring x is equivalent to swapping hex1/hex2, so parity survives.

    Args:
        side: "right" or "left".
        offset: Half-gap between the two eyes along x.

    Returns:
        list[dict]: EYEMAP_ROWS' columns plus ``hex1b``/``hex2b``.
    """
    rows = []
    for row in EYEMAP_ROWS:
        x = row["hex2"] - row["hex1"]
        y = row["hex2"] + row["hex1"]
        x_b = (x + offset) if side == "right" else -(x + offset)
        new = dict(row)
        new["hex1b"] = (y - x_b) // 2
        new["hex2b"] = (y + x_b) // 2
        # mirror the left eye's viewing direction, as the real eyemaps do
        new["y"] = row["y"] if side == "right" else -row["y"]
        rows.append(new)
    return rows


def coords_of(rows: list[dict], binocular: bool = False) -> list[str]:
    """
    Render eyemap rows as the ``'x,y'`` index strings the plotters expect.

    Args:
        rows: Eyemap rows carrying ``hex1``/``hex2`` (and ``hex1b``/``hex2b``
            when ``binocular``).
        binocular: Read the binocular hex pair instead of the per-eye one.

    Returns:
        list[str]: One ``f'{hex2 - hex1},{hex2 + hex1}'`` string per row.
    """
    hex1, hex2 = ("hex1b", "hex2b") if binocular else ("hex1", "hex2")
    return [f"{r[hex2] - r[hex1]},{r[hex2] + r[hex1]}" for r in rows]


def write_eyemap_dir(
    directory: Path,
    suffix: str = ".csv",
    sides: tuple = ("right", "left"),
    rows_by_side=None,
) -> dict:
    """
    Write per-hemisphere eyemap files into ``directory``, archive-style.

    This is the layout ``eyemap_dir`` expects: one file per hemisphere, named
    after it, so ``which_eye`` selects a file.

    Args:
        directory: Destination directory; created if absent.
        suffix: ``.csv``, ``.xls`` or ``.xlsx``.
        sides: Which hemisphere files to write.
        rows_by_side: Optional ``side -> rows`` callable, defaulting to
            ``binocular_rows``.

    Returns:
        dict: side -> rows written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    build = rows_by_side or binocular_rows
    rows = {}
    for side in sides:
        rows[side] = build(side)
        path = directory / f"{side}{suffix}"
        if suffix in (".xlsx", ".xls"):
            pd.DataFrame(rows[side]).to_excel(path, index=False)
        else:
            pd.DataFrame(rows[side]).to_csv(path, index=False)
    return rows


class TestEyemapDir(unittest.TestCase):
    """`eyemap_dir` is a dataset folder holding one file per hemisphere."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.csv_dir = self.root / "csv_dataset"
        self.xlsx_dir = self.root / "xlsx_dataset"
        write_eyemap_dir(self.csv_dir)
        write_eyemap_dir(self.xlsx_dir, suffix=".xlsx")

        # data index format matches "x,y" = (hex2 - hex1, hex2 + hex1), i.e. the
        # per-eye frame, so it aligns with EYEMAP_ROWS.
        self.data = pd.Series([1.0, 2.0, 3.0], index=["0,0", "1,1", "-1,-1"])

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_hex_heatmap_with_eyemap_dir(self):
        fig = hex_heatmap(self.data, eyemap_dir=str(self.csv_dir))
        self.assertIsInstance(fig, go.Figure)

    def test_hex_heatmap_with_eyemap_dir_xlsx(self):
        fig = hex_heatmap(self.data, eyemap_dir=str(self.xlsx_dir))
        self.assertIsInstance(fig, go.Figure)

    def test_plot_mollweide_projection_with_eyemap_dir(self):
        fig = plot_mollweide_projection(self.data, eyemap_dir=str(self.csv_dir))
        self.assertIsInstance(fig, go.Figure)

    def test_plot_mollweide_projection_with_eyemap_dir_xlsx(self):
        fig = plot_mollweide_projection(self.data, eyemap_dir=str(self.xlsx_dir))
        self.assertIsInstance(fig, go.Figure)

    def test_accepts_a_path_object(self):
        fig = hex_heatmap(self.data, eyemap_dir=self.csv_dir)
        self.assertIsInstance(fig, go.Figure)

    def test_defaults_to_the_right_eye(self):
        """No which_eye reads right.<ext> in the per-eye frame."""
        default = hex_heatmap(self.data, eyemap_dir=str(self.csv_dir))
        explicit = hex_heatmap(
            self.data, eyemap_dir=str(self.csv_dir), which_eye="right"
        )
        self.assertListEqual(list(default.data[0].x), list(explicit.data[0].x))
        # the right file's own hex1/hex2 lattice, untouched by hex1b/hex2b
        self.assertListEqual(
            sorted(default.data[0].x),
            sorted(r["hex2"] - r["hex1"] for r in EYEMAP_ROWS),
        )

    def test_prefers_xlsx_over_csv(self):
        """A dataset dir holding both formats reads the xlsx."""
        mixed = self.root / "mixed"
        write_eyemap_dir(mixed, suffix=".xlsx")
        # a decoy csv describing a different lattice
        decoy = [dict(r, hex1=r["hex1"] + 50, hex2=r["hex2"] + 50) for r in EYEMAP_ROWS]
        pd.DataFrame(decoy).to_csv(mixed / "right.csv", index=False)

        fig = hex_heatmap(self.data, eyemap_dir=str(mixed))
        self.assertListEqual(
            sorted(fig.data[0].y), sorted(r["hex2"] + r["hex1"] for r in EYEMAP_ROWS)
        )

    def test_missing_columns_raises(self):
        bad = self.root / "bad"
        bad.mkdir()
        pd.DataFrame([{"p": 0, "x": 0.0}]).to_csv(bad / "right.csv", index=False)

        for plot_fn in (hex_heatmap, plot_mollweide_projection):
            with self.subTest(plot_fn=plot_fn.__name__):
                with self.assertRaises(ValueError):
                    plot_fn(self.data, eyemap_dir=str(bad))

    def test_missing_hemisphere_raises(self):
        one_eye = self.root / "one_eye"
        write_eyemap_dir(one_eye, sides=("right",))
        with self.assertRaises(FileNotFoundError):
            hex_heatmap(self.data, eyemap_dir=str(one_eye), which_eye="left")

    def test_one_eye_ignores_the_absent_other_file(self):
        """which_eye='right' only ever opens right.<ext>."""
        one_eye = self.root / "right_only"
        write_eyemap_dir(one_eye, sides=("right",))
        fig = hex_heatmap(self.data, eyemap_dir=str(one_eye), which_eye="right")
        self.assertEqual(len(fig.data[0].x), len(EYEMAP_ROWS))

    def test_nonexistent_dir_raises(self):
        with self.assertRaises(NotADirectoryError):
            hex_heatmap(self.data, eyemap_dir=str(self.root / "nope"))

    def test_a_file_instead_of_a_dir_raises(self):
        with self.assertRaises(NotADirectoryError) as ctx:
            hex_heatmap(self.data, eyemap_dir=str(self.csv_dir / "right.csv"))
        self.assertIn("not a directory", str(ctx.exception))

    def test_empty_dir_raises(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            hex_heatmap(self.data, eyemap_dir=str(empty))

    def test_ignores_pq_columns(self):
        """hex1/hex2 are authoritative; a stale p/q must not move the lattice."""
        stale = self.root / "stale_pq"
        write_eyemap_dir(
            stale,
            rows_by_side=lambda side: [
                dict(r, p=r["p"] + 100, q=r["q"] - 7) for r in binocular_rows(side)
            ],
        )
        reference = hex_heatmap(self.data, eyemap_dir=str(self.csv_dir))
        fig = hex_heatmap(self.data, eyemap_dir=str(stale))
        self.assertListEqual(list(fig.data[0].x), list(reference.data[0].x))
        self.assertListEqual(list(fig.data[0].y), list(reference.data[0].y))

    def test_equal_aspect_is_default(self):
        fig = hex_heatmap(self.data, eyemap_dir=str(self.csv_dir))
        self.assertEqual(fig.layout.xaxis.scaleanchor, "y")
        self.assertAlmostEqual(fig.layout.xaxis.scaleratio, math.sqrt(3))

    def test_equal_aspect_off(self):
        fig = hex_heatmap(
            self.data, eyemap_dir=str(self.csv_dir), equal_aspect=False
        )
        self.assertIsNone(fig.layout.xaxis.scaleanchor)


class TestAutoFigWidth(unittest.TestCase):
    """hex_heatmap sizes itself to the lattice when given an eyemap_dir."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.data = pd.Series([1.0], index=["0,0"])

    def tearDown(self):
        self.tmpdir.cleanup()

    def _eyemap_spanning(self, x_min, x_max, y_min, y_max, name):
        """
        Write a right-eye eyemap whose lattice attains exactly the given extent.

        Lattice sites need even ``x + y``, and the extreme x and extreme y need
        not meet at one site (they do not on the real eyemaps), so each extreme
        gets its own row with the other coordinate nudged to keep parity.
        """
        pairs = [
            (x_min, y_min if (y_min - x_min) % 2 == 0 else y_min + 1),
            (x_max, y_min if (y_min - x_max) % 2 == 0 else y_min + 1),
            (x_min if (y_min - x_min) % 2 == 0 else x_min + 1, y_min),
            (x_min if (y_max - x_min) % 2 == 0 else x_min + 1, y_max),
        ]
        rows = [
            {"hex1": (y - x) // 2, "hex2": (y + x) // 2, "x": 0.0, "y": 0.0, "z": 1.0}
            for x, y in pairs
        ]
        frame = pd.DataFrame(rows)
        lattice_x = frame["hex2"] - frame["hex1"]
        lattice_y = frame["hex2"] + frame["hex1"]
        assert (lattice_x.min(), lattice_x.max()) == (x_min, x_max)
        assert (lattice_y.min(), lattice_y.max()) == (y_min, y_max)

        directory = Path(self.tmpdir.name) / name
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / "right.csv", index=False)
        return directory

    def _width_mm(self, fig):
        return round(fig.layout.width * MM_PER_PX)

    def _heatmap(self, directory, **kwargs):
        # off-lattice points are irrelevant here; the extent is what is tested
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return hex_heatmap(self.data, eyemap_dir=str(directory), **kwargs)

    def test_reproduces_package_default_for_one_medulla(self):
        """The mcns lattice extent must land back on the old 260/206 mm defaults."""
        directory = self._eyemap_spanning(-15, 18, 7, 70, "mcns_extent")
        self.assertEqual(self._width_mm(self._heatmap(directory)), 259)
        self.assertEqual(
            self._width_mm(self._heatmap(directory, colorbar=False)), 205
        )

    def test_widens_for_a_two_eye_extent(self):
        directory = self._eyemap_spanning(-38, 38, 7, 71, "two_eye_extent")
        self.assertEqual(self._width_mm(self._heatmap(directory)), 504)

    def test_follows_the_hex_aspect_relation(self):
        x_min, x_max, y_min, y_max = -38, 38, 7, 71
        directory = self._eyemap_spanning(x_min, x_max, y_min, y_max, "aspect")
        fig = self._heatmap(directory, colorbar=False)
        expected = 220 * ((x_max - x_min) + 2) / (((y_max - y_min) + 2) / math.sqrt(3))
        self.assertEqual(self._width_mm(fig), round(expected))

    def test_scales_with_fig_height(self):
        directory = self._eyemap_spanning(-38, 38, 7, 71, "half_height")
        fig = self._heatmap(directory, sizing={"fig_height": 110})
        # the colorbar's 54 mm is a fixed physical width, so it does not halve
        self.assertEqual(self._width_mm(fig), round((504 - CBAR_MM) / 2 + CBAR_MM))

    def test_explicit_fig_width_wins(self):
        directory = self._eyemap_spanning(-38, 38, 7, 71, "explicit")
        fig = self._heatmap(directory, sizing={"fig_width": 300})
        self.assertEqual(self._width_mm(fig), 300)

    def test_bundled_dataset_keeps_its_default(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = hex_heatmap(self.data, dataset="mcns_right")
        self.assertEqual(self._width_mm(fig), 260)

    def test_two_eye_directory_spans_both_blocks(self):
        """which_eye='both' concatenates the hemispheres, widening the figure."""
        directory = Path(self.tmpdir.name) / "both_eyes"
        rows = write_eyemap_dir(directory)
        data = pd.Series(
            [1.0] * 2 * len(EYEMAP_ROWS),
            index=coords_of(rows["right"], binocular=True)
            + coords_of(rows["left"], binocular=True),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            two_eye = hex_heatmap(data, eyemap_dir=str(directory), which_eye="both")
            bundled = hex_heatmap(data, dataset="mcns_right")
        self.assertGreater(self._width_mm(two_eye), self._width_mm(bundled))


class TestWhichEye(unittest.TestCase):
    """`which_eye` picks the file(s) and, with them, the hex frame."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.eyemap_dir = Path(self.tmpdir.name) / "eyemap_dataset"
        self.rows = write_eyemap_dir(self.eyemap_dir)
        self.per_eye = len(EYEMAP_ROWS)

        self.binocular_data = pd.Series(
            range(2 * self.per_eye),
            index=coords_of(self.rows["right"], binocular=True)
            + coords_of(self.rows["left"], binocular=True),
            dtype=float,
        )
        self.per_eye_data = pd.Series(
            range(self.per_eye), index=coords_of(EYEMAP_ROWS), dtype=float
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_both_concatenates_the_hemispheres(self):
        fig = hex_heatmap(
            self.binocular_data, eyemap_dir=str(self.eyemap_dir), which_eye="both"
        )
        self.assertEqual(len(fig.data[0].x), 2 * self.per_eye)
        self.assertEqual(len(fig.data[1].x), 2 * self.per_eye)

    def test_both_separates_the_eyes(self):
        fig = hex_heatmap(
            self.binocular_data, eyemap_dir=str(self.eyemap_dir), which_eye="both"
        )
        expected = sorted(
            r["hex2b"] - r["hex1b"]
            for side_rows in self.rows.values()
            for r in side_rows
        )
        self.assertListEqual(sorted(fig.data[0].x), expected)
        # no hexagon straddles the mid-line, and each eye keeps its own sign
        self.assertTrue(all(x != 0 for x in fig.data[0].x))

    def test_one_eye_uses_the_per_eye_frame(self):
        for which_eye in ("right", "left"):
            with self.subTest(which_eye=which_eye):
                fig = hex_heatmap(
                    self.per_eye_data,
                    eyemap_dir=str(self.eyemap_dir),
                    which_eye=which_eye,
                )
                self.assertEqual(len(fig.data[0].x), self.per_eye)
                # both hemisphere files share one hex1/hex2 lattice
                self.assertListEqual(
                    sorted(fig.data[0].x),
                    sorted(r["hex2"] - r["hex1"] for r in EYEMAP_ROWS),
                )

    def test_the_two_eyes_differ_only_by_viewing_direction(self):
        """Same hex1/hex2 lattice, but mirrored across the mid-line on the sphere."""
        right, left = (
            plot_mollweide_projection(
                self.per_eye_data, eyemap_dir=str(self.eyemap_dir), which_eye=eye
            )
            for eye in ("right", "left")
        )
        self.assertEqual(len(right.data[-1].x), len(left.data[-1].x))
        # the fixture mirrors the left eye's y component, which is azimuth --
        # Mollweide x -- so elevation (y) is deliberately unchanged
        self.assertNotEqual(list(right.data[-1].x), list(left.data[-1].x))
        self.assertListEqual(list(right.data[-1].y), list(left.data[-1].y))

    def test_unknown_which_eye_raises(self):
        for plot_fn in (hex_heatmap, plot_mollweide_projection):
            with self.subTest(plot_fn=plot_fn.__name__):
                with self.assertRaises(ValueError) as ctx:
                    plot_fn(
                        self.per_eye_data,
                        eyemap_dir=str(self.eyemap_dir),
                        which_eye="middle",
                    )
                self.assertIn("which_eye must be one of", str(ctx.exception))

    def test_bundled_datasets_are_right_eye_only(self):
        for plot_fn in (hex_heatmap, plot_mollweide_projection):
            for which_eye in ("left", "both"):
                with self.subTest(plot_fn=plot_fn.__name__, which_eye=which_eye):
                    with self.assertRaises(ValueError) as ctx:
                        plot_fn(self.per_eye_data, which_eye=which_eye)
                    self.assertIn("eyemap_dir", str(ctx.exception))

    def test_bundled_datasets_still_work_by_default(self):
        for plot_fn in (hex_heatmap, plot_mollweide_projection):
            with self.subTest(plot_fn=plot_fn.__name__):
                with warnings.catch_warnings():
                    # the tiny fixture coords are not on the bundled lattice
                    warnings.simplefilter("ignore")
                    self.assertIsInstance(plot_fn(self.per_eye_data), go.Figure)

    def test_bundled_mollweide_places_its_own_lattice(self):
        """Regression: the bundled lattice must carry hex1/hex2, not p/q."""
        bundled = load_dataset("Zhao2024")
        self.assertTrue({"hex1", "hex2"} <= set(bundled.columns), list(bundled.columns))
        hex1, hex2 = bundled["hex1"], bundled["hex2"]
        coords = ((hex2 - hex1).astype(str) + "," + (hex2 + hex1).astype(str)).unique()
        data = pd.Series(1.0, index=coords)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = plot_mollweide_projection(data)
        self.assertEqual(
            [str(w.message) for w in caught if w.category is UserWarning], []
        )
        self.assertEqual(len(fig.data[-1].x), len(data))

    def test_adding_the_columns_leaves_the_per_eye_frame_alone(self):
        """hex1b/hex2b are additive: hex1/hex2 readers see no change."""
        bare = Path(self.tmpdir.name) / "bare"
        write_eyemap_dir(bare, rows_by_side=lambda side: list(EYEMAP_ROWS))
        reference = hex_heatmap(self.per_eye_data, eyemap_dir=str(bare))
        fig = hex_heatmap(self.per_eye_data, eyemap_dir=str(self.eyemap_dir))
        self.assertListEqual(list(fig.data[0].x), list(reference.data[0].x))
        self.assertListEqual(list(fig.data[0].y), list(reference.data[0].y))

    def test_both_needs_the_binocular_columns(self):
        bare = Path(self.tmpdir.name) / "bare_both"
        write_eyemap_dir(bare, rows_by_side=lambda side: list(EYEMAP_ROWS))
        for plot_fn in (hex_heatmap, plot_mollweide_projection):
            with self.subTest(plot_fn=plot_fn.__name__):
                with self.assertRaises(ValueError) as ctx:
                    plot_fn(
                        self.binocular_data, eyemap_dir=str(bare), which_eye="both"
                    )
                message = str(ctx.exception)
                self.assertIn("hex1b", message)
                self.assertIn("download_eyemaps", message)


class TestFrameMismatch(unittest.TestCase):
    """Plotting an index in the wrong hex frame must not pass silently."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.eyemap_dir = Path(self.tmpdir.name) / "eyemap_dataset"
        self.rows = write_eyemap_dir(self.eyemap_dir)
        self.binocular_data = pd.Series(
            range(2 * len(EYEMAP_ROWS)),
            index=coords_of(self.rows["right"], binocular=True)
            + coords_of(self.rows["left"], binocular=True),
            dtype=float,
        )
        self.per_eye_data = pd.Series(
            range(len(EYEMAP_ROWS)), index=coords_of(EYEMAP_ROWS), dtype=float
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def _warnings_from(self, plot_fn, data, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_fn(data, eyemap_dir=str(self.eyemap_dir), **kwargs)
        return [str(w.message) for w in caught if w.category is UserWarning]

    def test_matching_frames_are_silent(self):
        cases = (
            (self.per_eye_data, "right"),
            (self.per_eye_data, "left"),
            (self.binocular_data, "both"),
        )
        for data, which_eye in cases:
            with self.subTest(which_eye=which_eye):
                self.assertEqual(
                    self._warnings_from(hex_heatmap, data, which_eye=which_eye), []
                )

    def test_binocular_index_on_a_one_eye_lattice_warns(self):
        n = len(self.binocular_data)
        messages = self._warnings_from(
            hex_heatmap, self.binocular_data, which_eye="right"
        )
        self.assertEqual(len(messages), 1)
        self.assertIn(f"{n} of {n} plotted columns", messages[0])
        self.assertIn("other hex frame", messages[0])

    def test_per_eye_index_on_the_two_eye_lattice_warns(self):
        n = len(self.per_eye_data)
        messages = self._warnings_from(hex_heatmap, self.per_eye_data, which_eye="both")
        self.assertEqual(len(messages), 1)
        self.assertIn(f"{n} of {n} plotted columns", messages[0])

    def test_a_single_absent_column_is_counted(self):
        """Rim columns are reported too -- the count is what tells them apart."""
        # off the lattice but still even-parity, as a real hex column must be
        off_lattice = max(r["hex2"] - r["hex1"] for r in EYEMAP_ROWS) + 1
        rim = pd.concat(
            [self.per_eye_data, pd.Series([9.0], index=[f"{off_lattice},0"])]
        )
        messages = self._warnings_from(hex_heatmap, rim, which_eye="right")
        self.assertEqual(len(messages), 1)
        self.assertIn(f"1 of {len(rim)} plotted columns", messages[0])

    def test_hex_heatmap_and_mollweide_agree_on_what_is_unplaced(self):
        """Both functions must count the same columns as off-lattice."""
        # off the lattice but still even-parity, as a real hex column must be
        off_lattice = max(r["hex2"] - r["hex1"] for r in EYEMAP_ROWS) + 1
        rim = pd.concat(
            [self.per_eye_data, pd.Series([9.0], index=[f"{off_lattice},0"])]
        )
        heat = self._warnings_from(hex_heatmap, rim, which_eye="right")
        moll = self._warnings_from(plot_mollweide_projection, rim, which_eye="right")
        self.assertTrue(
            any(f"1 of {len(rim)} plotted columns" in m for m in heat), heat
        )
        self.assertTrue(any(f"1 of {len(rim)} rows" in m for m in moll), moll)

    def test_mollweide_reports_the_mismatch_as_unplaced_rows(self):
        messages = self._warnings_from(
            plot_mollweide_projection, self.binocular_data, which_eye="right"
        )
        self.assertEqual(len(messages), 1)
        self.assertIn("no matching hex1/hex2", messages[0])
        self.assertIn("other hex frame", messages[0])


class TestMollweideUnplacedRows(unittest.TestCase):
    """Rows with no lattice hex cannot be projected, and must be reported."""

    def setUp(self):
        self.tmpdir = TemporaryDirectory()
        self.eyemap_dir = Path(self.tmpdir.name) / "eyemap_dataset"
        write_eyemap_dir(self.eyemap_dir)
        self.on_lattice = pd.Series(
            range(len(EYEMAP_ROWS)), index=coords_of(EYEMAP_ROWS), dtype=float
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_no_warning_when_every_row_is_placed(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_mollweide_projection(self.on_lattice, eyemap_dir=str(self.eyemap_dir))
        self.assertEqual([w for w in caught if w.category is UserWarning], [])

    def test_warns_and_counts_unplaced_rows(self):
        data = pd.concat(
            [self.on_lattice, pd.Series([9.0, 9.0], index=["40,40", "42,40"])]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plot_mollweide_projection(data, eyemap_dir=str(self.eyemap_dir))
        messages = [str(w.message) for w in caught if w.category is UserWarning]
        self.assertEqual(len(messages), 1)
        self.assertIn(f"2 of {len(data)} rows", messages[0])
