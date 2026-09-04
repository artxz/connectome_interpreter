import io
import pkgutil
import os
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

DATA_SOURCES: dict[str, str] = {
    "DoOR_adult": "data/DoOR/processed_door_adult.csv",
    "DoOR_adult_sfr_subtracted": "data/DoOR/processed_door_adult_sfr_subtracted.csv",
    "Dweck_adult_chem": "data/Dweck2018/adult_chem2glom.csv",
    "Dweck_adult_fruit": "data/Dweck2018/adult_fruit2glom.csv",
    "Dweck_larva_chem": "data/Dweck2018/larva_chem2or.csv",
    "Dweck_larva_fruit": "data/Dweck2018/larva_fruit2or.csv",
    "Nern2024": "data/Nern2024/ME-columnar-cells-hex-location.csv",
    "Matsliah2024": "data/Matsliah2024/fafb_right_vis_cols.csv",
    "Badel2016_PN": "data/Badel2016/Badel2016.csv",
    "Zhao2024": "data/Zhao2024/ucl_hex_right_20240701_tomale.csv",
    "Hallem2006": "data/Hallem_Carlson_2006/odour_response_s1_tidy.csv",
    "Hallem2006_dilution": "data/Hallem_Carlson_2006/odour_dilution_response_s2_tidy.csv",
    "Hallem2006_time": "data/Hallem_Carlson_2006/odour_time_response_s3_tidy.csv",
    "Knaden2012_odour_valence": "data/Knaden2012/Knaden2012_odour_valence.csv",
}


def load_dataset(dataset: str) -> pd.DataFrame:
    """
    Load the dataset from the package data folder. These datasets have been
    preprocessed to work with connectomics data. The preprocessing scripts are
    in this repository: https://github.com/YijieYin/interpret_connectome.

    Args:
        dataset: (str) The name of the dataset to load. Options are:

            - 'DoOR_adult': mapping from glomeruli to chemicals, from Munch and Galizia DoOR dataset (https://www.nature.com/articles/srep21841), a composite of multiple studies and their own data. When it's their own data (not specified), odour concentration is 10^-2. Ca imaging.
            - 'DoOR_adult_sfr_subtracted': mapping from glomeruli to chemicals, with spontaneous firing rate subtracted. There are therefore negative values.
            - 'Dweck_adult_chem': mapping from glomeruli to chemicals extracted from fruits, from Dweck et al. 2018 (https://www.cell.com/cell-reports/abstract/S2211-1247(18)30663-6). Normalised maximum frequency (Hz) responses to  10^-4 concentration of synthetic standards of the active compounds. Firing rates normalised to between 0 and 1. Electrophysiology data.
            - 'Dweck_adult_fruit': number of compounds in a fruit that activated a glomerulus, from Dweck et al. 2018. Not normalised because compound count is not response magnitude.
            - 'Dweck_larva_chem': mapping from olfactory receptors to chemicals, from Dweck et al. 2018. Normalised maximum frequency (Hz) responses to  10^-4 concentration of synthetic standards of the active compounds. Firing rates normalised to between 0 and 1.
            - 'Dweck_larva_fruit': number of compounds in a fruit that activated a receptor, from Dweck et al. 2018. Not normalised because compound count is not response magnitude.
            - 'Nern2024': columnar coordinates of individual cells from a collection of columnar cell types within the medulla of the right optic lobe, from Nern et al. 2024 (https://www.biorxiv.org/content/10.1101/2024.04.16.589741v2).
            - 'Matsliah2024': columnar coordinates of individual cells from a collection of columnar cell types in the right optic lobe from FAFB, from Matsliah et al. 2024 (https://www.nature.com/articles/s41586-024-07981-1).
            - 'Badel2016_PN': mapping from olfactory projection neurons to odours, from Badel et al. 2016 (https://www.cell.com/neuron/fulltext/S0896-6273(16)30201-X). Odour dilution is 10^-2 unless otherwise specified. Ca imaging.
            - 'Zhao2024': mapping from hexagonal coordinates to 3D coordinates, update from Zhao et al. 2022 (https://www.biorxiv.org/content/10.1101/2022.12.14.520178v1).
            - 'Hallem2006': mapping from glomeruli to chemicals, from Hallem and Carlson 2006 (https://www.cell.com/cell/abstract/S0092-8674(06)00363-1). Odour dilution is 10^-2 unless otherwise specified. Electrophysiology data.
            - 'Hallem2006_dilution': mapping from glomeruli to chemicals across dilution rates, from Hallem and Carlson 2006.
            - 'Hallem2006_time': response of glomeruli to odours, across timepoints, from Hallem and Carlson 2006.
            - 'Knaden2012_odour_valence': behavioural valence of odours, from Knaden et al. 2012 (https://www.sciencedirect.com/science/article/pii/S2211124712000733).

    Returns:
        pd.DataFrame:
            The dataset as a pandas DataFrame. For the adult, the glomeruli are in the
            rows. For the larva, receptors are in the rows.
    """

    try:
        data = pkgutil.get_data("connectome_interpreter", DATA_SOURCES[dataset])
    except KeyError as exc:
        raise ValueError(
            "Dataset not recognized. Please choose from {}".format(
                list(DATA_SOURCES.keys())
            )
        ) from exc

    return pd.read_csv(io.BytesIO(data), index_col=0)


# The per-eye hex ids every eyemap carries, and the binocular re-encoding that
# the eyemap archive's `download_eyemaps.py` appends alongside them. Both optic
# lobes are numbered in the same hex1/hex2 frame, so a left and a right column
# can share an id pair -- hex1b/hex2b move the eyes into disjoint blocks, which
# is the only way to draw them together.
_HEX_COLS = ("hex1", "hex2")
_HEX_COLS_BINOCULAR = ("hex1b", "hex2b")

# The archive stores one hemisphere per file, named after the hemisphere.
_EYE_CHOICES = ("right", "left", "both")
_EYEMAP_SUFFIXES = (".xlsx", ".xls", ".csv")


def _hex_columns(which_eye: str) -> tuple:
    """
    The pair of hex id columns to read for a given ``which_eye``.

    One eye is drawn in its own per-eye numbering, ``hex1``/``hex2``. Both eyes
    at once needs ``hex1b``/``hex2b``, because the per-eye ids collide between
    the hemispheres. So the frame follows from ``which_eye`` — and the plotted
    data's index has to be in the matching frame.

    Args:
        which_eye: One of ``'right'``, ``'left'``, ``'both'``.

    Returns:
        tuple: ``(hex1_col, hex2_col)``.

    Raises:
        ValueError: If ``which_eye`` is not one of the three choices.
    """
    if which_eye not in _EYE_CHOICES:
        raise ValueError(f"which_eye must be one of {_EYE_CHOICES}, got {which_eye!r}.")
    return _HEX_COLS_BINOCULAR if which_eye == "both" else _HEX_COLS


def _eyemap_files(eyemap_dir, which_eye: str) -> list:
    """
    The eyemap file(s) inside ``eyemap_dir`` that ``which_eye`` asks for.

    The archive writes one file per hemisphere, named after it, so selecting an
    eye is selecting a file: ``'right'`` reads ``right.xlsx``, ``'both'`` reads
    both and concatenates them.

    Args:
        eyemap_dir: Dataset directory holding ``right.xlsx``/``left.xlsx``, as
            written by the archive's ``download_eyemaps.py``. ``.xls`` and
            ``.csv`` are accepted too, in that order of preference.
        which_eye: One of ``'right'``, ``'left'``, ``'both'``.

    Returns:
        list: ``pathlib.Path`` objects, right eye first.

    Raises:
        NotADirectoryError: If ``eyemap_dir`` is not a directory.
        FileNotFoundError: If a hemisphere ``which_eye`` needs has no file.
    """
    directory = Path(eyemap_dir)
    if not directory.is_dir():
        raise NotADirectoryError(
            f"eyemap_dir {directory} is not a directory. Pass the dataset folder "
            "holding one eyemap file per hemisphere (e.g. "
            "params/eyemaps/eyemap_mcns_f20240701/, holding right.xlsx and "
            "left.xlsx), as written by the eyemap archive's download_eyemaps.py."
        )

    wanted = ("right", "left") if which_eye == "both" else (which_eye,)
    files = []
    for stem in wanted:
        for suffix in _EYEMAP_SUFFIXES:
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                files.append(candidate)
                break
        else:
            raise FileNotFoundError(
                f"eyemap_dir {directory} holds no {stem} eyemap (looked for "
                f"{stem} with any of {_EYEMAP_SUFFIXES}), which "
                f"which_eye={which_eye!r} needs. Download it from "
                "https://artxz.github.io/eyemap-archive/."
            )
    return files


def _load_local_eyemap(
    eyemap_dir, which_eye: str, extra_cols: tuple = ()
) -> pd.DataFrame:
    """
    Load a local eyemap directory, in place of a bundled ``dataset``.

    The hex id columns are the authoritative lattice columns: the hexagon
    positions are always derived from them, as ``x = hex2 - hex1``,
    ``y = hex2 + hex1`` — or the ``hex1b``/``hex2b`` equivalents when
    ``which_eye='both'``; see ``_hex_columns()``. Files downloaded from the
    eyemap archive also carry a ``p,q`` numbering (related by ``hex1 = q + 18``,
    ``hex2 = p + 19``), but it is **ignored** here.

    Args:
        eyemap_dir: Dataset directory holding one eyemap file per hemisphere;
            see ``_eyemap_files()``. Files downloaded via
            https://artxz.github.io/eyemap-archive/ already conform to the
            expected schema (columns
            ``hex1,hex2,hex1b,hex2b,x,y,z[,p,q,theta,phi,rootid]``).
        which_eye: One of ``'right'``, ``'left'``, ``'both'``.
        extra_cols: Column names required on top of the hex id pair.

    Returns:
        pd.DataFrame: The loaded eyemap table, hemispheres concatenated for
        ``which_eye='both'``.
    """
    required_cols = _hex_columns(which_eye) + tuple(extra_cols)
    frames = []
    for path in _eyemap_files(eyemap_dir, which_eye):
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        missing = set(required_cols) - set(df.columns)
        if missing:
            hint = ""
            if missing & set(_HEX_COLS_BINOCULAR):
                hint = (
                    f" The two-eye columns {'/'.join(_HEX_COLS_BINOCULAR)} are "
                    "added by the eyemap archive's download_eyemaps.py — re-run "
                    "it, or ask for a single eye, which is drawn from "
                    f"{'/'.join(_HEX_COLS)}."
                )
            raise ValueError(
                f"eyemap file {path} is missing required column(s): "
                f"{sorted(missing)}. Expected columns: {required_cols}. Files "
                "downloaded from https://artxz.github.io/eyemap-archive/ already "
                f"conform to this schema.{hint}"
            )
        frames.append(df)

    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


# Pixels per x-unit divided by pixels per y-unit for a regular hex lattice in
# these doubled coordinates. Neighbours sit at (dx, dy) = (+-1, +-1) and
# (0, +-2), so equal nearest-neighbour distance needs a^2 + b^2 = (2b)^2.
_HEX_ASPECT = np.sqrt(3)


def _warn_unplaced(
    x_vals, y_vals, background_hex: pd.DataFrame, which_eye: str
) -> None:
    """
    Warn about plotted columns that are not on the eyemap's lattice.

    They are still drawn, just with no background hexagon behind them, so
    without this they pass unremarked. The proportion says which of two causes
    it is: a few percent means columns present in the connectome but absent from
    the eyemap — rim columns with no fitted viewing direction — while most of
    them means the data's index is in the *other* hex frame, since ``which_eye``
    selects the hex id columns as well as the lattice.

    Matching is on the exact ``(x, y)`` site, which is what
    ``plot_mollweide_projection()``'s merge effectively does, so the two
    functions report the same columns as unplaced.

    Args:
        x_vals: Data x coordinates.
        y_vals: Data y coordinates.
        background_hex: The background lattice being drawn.
        which_eye: One of ``'right'``, ``'left'``, ``'both'``.
    """
    sites = set(
        zip(
            np.asarray(background_hex["x"], dtype=float).tolist(),
            np.asarray(background_hex["y"], dtype=float).tolist(),
        )
    )
    points = list(
        zip(
            np.asarray(x_vals, dtype=float).tolist(),
            np.asarray(y_vals, dtype=float).tolist(),
        )
    )
    n_unplaced = sum(1 for point in points if point not in sites)
    if not n_unplaced:
        return
    warnings.warn(
        f"{n_unplaced} of {len(points)} plotted columns are not on the "
        f"which_eye={which_eye!r} eyemap lattice, so they are drawn with no "
        "background hexagon. A few percent is expected: columns present in the "
        "connectome but absent from the eyemap, typically rim columns with no "
        "fitted viewing direction. Most of them instead means the index is in "
        f"the other hex frame -- which_eye='both' expects "
        f"{'/'.join(_HEX_COLS_BINOCULAR)} coordinates, one eye expects "
        f"{'/'.join(_HEX_COLS)}.",
        UserWarning,
        stacklevel=3,
    )


def map_to_experiment(df, dataset=None, custom_experiment=None):
    """
    Map the connectomics data to experimental data. For example, if odour1
    excites neuron1 0.5, and neuron2 0.6; both neuron1 and neuron2 output to
    neuron3 (0.7 and 0.8 respectively), then the output of neuron3 to odour1
    is 0.5*0.7 + 0.6*0.8 = 0.83. The result would only be 1 if a stimulus
    excites neurons 100%, and those neurons constitue 100% of the downstream
    neuron's input.

    Args:
        df (pd.DataFrame): The connectivity data. Standardised input (e.g. glomeruli,
            receptors) in rows, observations (target neurons) in columns.
        dataset (str): The name of the dataset to load. Options are:

            - 'DoOR_adult': mapping from glomeruli to chemicals, from Munch and Galizia DoOR dataset (https://www.nature.com/articles/srep21841), a composite of multiple studies and their own data. When it's their own data (not specified), odour concentration is 10^-2. Ca imaging.
            - 'DoOR_adult_sfr_subtracted': mapping from glomeruli to chemicals, with spontaneous firing rate subtracted. There are therefore negative values.
            - 'Dweck_adult_chem': mapping from glomeruli to chemicals extracted from fruits, from Dweck et al. 2018 (https://www.cell.com/cell-reports/abstract/S2211-1247(18)30663-6). Normalised maximum frequency (Hz) responses to  10^-4 concentration of synthetic standards of the active compounds. Firing rates normalised to between 0 and 1. Electrophysiology data.
            - 'Dweck_adult_fruit': number of compounds in a fruit that activated a glomerulus, from Dweck et al. 2018. Not normalised because compound count is not response magnitude.
            - 'Dweck_larva_chem': mapping from olfactory receptors to chemicals, from Dweck et al. 2018. Normalised maximum frequency (Hz) responses to  10^-4 concentration of synthetic standards of the active compounds. Firing rates normalised to between 0 and 1.
            - 'Dweck_larva_fruit': number of compounds in a fruit that activated a receptor, from Dweck et al. 2018. Not normalised because compound count is not response magnitude.
            - 'Nern2024': columnar coordinates of individual cells from a collection of columnar cell types within the medulla of the right optic lobe, from Nern et al. 2024.
            - 'Badel2016_PN': mapping from olfactory projection neurons to odours, from Badel et al. 2016 (https://www.cell.com/neuron/fulltext/S0896-6273(16)30201-X). Odour dilution is 10^-2 unless otherwise specified. Ca imaging.
            - 'Hallem2006': mapping from glomeruli to chemicals, from Hallem and Carlson 2006 (https://www.cell.com/cell/abstract/S0092-8674(06)00363-1). Odour dilution is 10^-2 unless otherwise specified. Electrophysiology data.
            - 'Hallem2006_dilution': mapping from glomeruli to chemicals across dilution rates, from Hallem and Carlson 2006.

        custom_experiment (pd.DataFrame): A custom experimental dataset to compare the
            connectomics data to. The row indices of this dataframe must match the row
            indices of df. They are the units of comparison (e.g. glomeruli).

    Returns:
        pd.DataFrame:
            The similarity between the connectomics data and the experimental data. Rows
            are neurons, columns are external stimulus.
    """

    # try:
    #     from sklearn.metrics.pairwise import cosine_similarity
    # except ImportError as e:
    #     raise ImportError(
    #         "To use this function, please install scikit-learn. You can
    # install it with 'pip install scikit-learn'.") from e
    if dataset is not None and custom_experiment is not None:
        raise ValueError(
            "Please provide either a dataset or a custom_experiment, not both."
        )
    if dataset is None and custom_experiment is None:
        raise ValueError("Please provide either a dataset or a custom_experiment.")
    if dataset is not None:
        data = load_dataset(dataset)
    else:
        data = custom_experiment

    # take the intersection of glomeruli
    data = data[data.index.isin(df.index)]
    df_intersect = df[df.index.isin(data.index)]
    df_intersect = df_intersect.reindex(data.index)

    # multiply the correpsonding values using matmul
    target2chem = np.dot(df_intersect.values.T, data.values)
    # Assign appropriate column names
    target2chem = pd.DataFrame(
        target2chem, index=df_intersect.columns, columns=data.columns
    )
    return target2chem


def hex_heatmap(
    df: pd.Series | pd.DataFrame,
    style: Optional[dict] = None,
    sizing: Optional[dict] = None,
    dpi: int = 72,
    custom_colorscale: Optional[Union[list, str]] = None,
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
    dataset: Optional[str] = "mcns_right",
    eyemap_dir: Optional[str] = None,
    which_eye: str = "right",
    equal_aspect: bool = True,
    value_name: str = "weight",
    colorbar: bool = True,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Generate a hexagonal heat map plot of the data. The index of the data
    should be formatted as strings of the form '-12,34', where the first
    number is the x-coordinate and the second number is the y-coordinate.

    Args:
        df (pd.Series | pd.DataFrame): The data to plot. If a Series, it will be plotted
            as a single trace. If a DataFrame, each column will generate a separate
            frame in the plot.
        style (Optional[dict]): Dict containing styling formatting variables. Possible
            keys are:

                - 'font_type': str, default='arial'
                - 'linecolor': str, default='black'
                - 'papercolor': str, default='rgba(255,255,255,255)' (white)

        sizing (Optional[dict]): Dict containing size formatting variables. Possible
            keys are:

                - 'fig_width': int, default=260 (mm). With ``eyemap_dir``, defaults instead to the width that makes the lattice's own extent fill the figure, so markers tile without gaps whatever the lattice covers (~260 mm for one medulla, ~504 mm for two eyes side by side).
                - 'fig_height': int, default=220 (mm)
                - 'fig_margin': int, default=0 (mm)
                - 'fsize_ticks_pt': int, default=20 (points)
                - 'fsize_title_pt': int, default=20 (points) - colorbar title font size
                - 'fsize_plot_title_pt': int, default=24 (points) - plot title font size
                - 'title_margin': int, default=50 (pixels) - top margin when title is present
                - 'markersize': int, default=18 if dataset='mcns_right', 20 if dataset='fafb_right'
                - 'ticklen': int, default=15
                - 'tickwidth': int, default=5
                - 'axislinewidth': int, default=3
                - 'markerlinewidth': int, default=0.9
                - 'cbar_thickness': int, default=20
                - 'cbar_len': float, default=0.75

        dpi (int): Dots per inch for the output figure. Standard is 72 for screen/SVG/PDF.
            Use higher values (e.g., 300) for print-quality output.
        custom_colorscale (Optional[Union[list, str]]): Custom colorscale for the
            heatmap. If None, defaults to white-to-blue colorscale [[0, "rgb(255, 255,
            255)"], [1, "rgb(0, 20, 200)"]].
        global_min (Optional[float]): Global minimum value for the color scale. If None,
            the minimum value of the data is used but if that is negative, use 0.
        global_max (Optional[float]): Global maximum value for the color scale. If None,
            the maximum value of the data is used.
        dataset (str): Default='mcns_right'. The dataset to use for the hexagon
            locations. Options are:

                - 'mcns_right': columnar coordinates of individual cells from columnar cell types: L1, L2, L3, L5, Mi1, Mi4, Mi9, C2, C3, Tm1, Tm2, Tm4, Tm9, Tm20, T1, within the medulla of the right optic lobe, from Nern et al. 2024.
                - 'fafb_right': columnar coordinates of individual cells from columnar cell types, in the right optic lobe of FAFB, from Matsliah et al. 2024.

        eyemap_dir (Optional[str]): Path to a **directory** holding one eyemap
            file per hemisphere — ``right.xlsx`` and ``left.xlsx``, as written by
            the eyemap archive's ``download_eyemaps.py`` (``.xls``/``.csv`` also
            accepted). ``which_eye`` picks which of them to read. When given,
            this takes precedence over ``dataset`` for the background hexagon
            lattice. Any ``p``/``q`` columns are ignored; see
            ``_load_local_eyemap()``.
        which_eye (str): Default='right'. Which eye to draw, one of 'right',
            'left', 'both'. This chooses both the file(s) read and the hex id
            columns they are read from, so ``df``'s index must be in the matching
            frame: one eye is drawn from that hemisphere's own ``hex1``/``hex2``,
            while 'both' is drawn from ``hex1b``/``hex2b`` — the same lattice
            sites re-encoded into disjoint per-eye blocks, since the two optic
            lobes share one ``hex1``/``hex2`` numbering and would otherwise
            collapse onto each other. Points landing off the resulting lattice
            are drawn but warned about, which is what catches a frame mismatch.
            Only 'right' is valid without ``eyemap_dir``: the bundled datasets
            cover the right optic lobe only.
        equal_aspect (bool): Default=True. Lock the x axis to sqrt(3) times the
            y axis scale via plotly's ``scaleanchor``/``scaleratio``, which is
            what makes the hexagons regular in these doubled coordinates
            (neighbours at (dx, dy) = (+-1, +-1) and (0, +-2)). With this on,
            hexagon shape no longer depends on the figure's aspect ratio - a
            mismatched ``fig_width``/``fig_height`` letterboxes instead of
            shearing. Set False to reproduce figures made before this option.
        title (Optional[str]): Title for the plot. If None, no title is displayed.

    Returns:
        fig : go.Figure
    """

    def bg_hex():
        """
        Generate a scatter plot of the background hexagons."
        """
        goscatter = go.Scatter(
            x=background_hex["x"],
            y=background_hex["y"],
            mode="markers",
            marker_symbol=symbol_number,
            marker={
                "size": sizing["markersize"],
                "color": "white",
                "line": {
                    "width": sizing["markerlinewidth"],
                    "color": "lightgrey",
                },
            },
            showlegend=False,
        )
        return goscatter

    def data_hex(aseries):
        """
        Generate a scatter plot of the data hexagons.""
        """
        marker_config = {
            "cmin": global_min,
            "cmax": global_max,
            "size": sizing["markersize"],
            "color": aseries.values,
            "line": {
                "width": sizing["markerlinewidth"],
                "color": "lightgrey",
            },
            "colorscale": custom_colorscale,
        }

        if colorbar:
            marker_config["colorbar"] = {
                "orientation": "v",
                "outlinecolor": style["linecolor"],
                "outlinewidth": sizing["axislinewidth"],
                "thickness": sizing["cbar_thickness"],
                "len": sizing["cbar_len"],
                "tickmode": "array",
                "ticklen": sizing["ticklen"],
                "tickwidth": sizing["tickwidth"],
                "tickcolor": style["linecolor"],
                "tickfont": {
                    "size": fsize_ticks_px,
                    "family": style["font_type"],
                    "color": style["linecolor"],
                },
                "tickformat": ".5f",
                "title": {
                    "font": {
                        "family": style["font_type"],
                        "size": fsize_title_px,
                        "color": style["linecolor"],
                    },
                    "side": "right",
                },
            }
        goscatter = go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker_symbol=symbol_number,
            customdata=np.stack([x_vals, y_vals, aseries.values], axis=-1),
            hovertemplate="x: %{customdata[0]}<br>y: %{customdata[1]}<br>%{text}: %{customdata[2]:.4f}",
            text=[value_name] * len(aseries),
            marker=marker_config,
            showlegend=False,
        )
        return goscatter

    # begin with removing nan from the index
    df = df[(df.index != "nan") & (~df.index.isnull())]

    # Default styling and sizing parameters to use if not specified.
    default_style = {
        "font_type": "arial",
        "markerlinecolor": "rgba(0,0,0,0)",  # transparent
        "linecolor": "black",
        "papercolor": "rgba(255,255,255,255)",
    }

    # load all hex coordinates. Done before the sizing defaults, which need the
    # lattice's extent.
    hex1_col, hex2_col = _hex_columns(which_eye)
    if eyemap_dir is not None:
        eyemap = _load_local_eyemap(eyemap_dir, which_eye)
        background_hex = pd.DataFrame(
            {
                "x": eyemap[hex2_col] - eyemap[hex1_col],
                "y": eyemap[hex2_col] + eyemap[hex1_col],
            }
        )
        markersize = 18
    elif which_eye != "right":
        raise ValueError(
            f"which_eye={which_eye!r} needs eyemap_dir: the bundled datasets "
            "('mcns_right', 'fafb_right') cover the right optic lobe only."
        )
    elif dataset == "mcns_right":
        background_hex = load_dataset("Nern2024")
        markersize = 18
    elif dataset == "fafb_right":
        background_hex = load_dataset("Matsliah2024")
        markersize = 20
    else:
        # raise error
        raise ValueError(
            "Dataset not recognized. Currently available datasets are 'mcns_right', "
            "'fafb_right'."
        )
    # only get the unique combination of 'x' and 'y' columns
    background_hex = background_hex.drop_duplicates(subset=["x", "y"])

    default_sizing = {
        # None with `eyemap_dir`: derived from the lattice extent below, once
        # any caller-supplied `fig_height` is known.
        "fig_width": None if eyemap_dir is not None else (260 if colorbar else 206),
        "fig_height": 220,  # units = mm
        "fig_margin": 0,
        "fsize_ticks_pt": 20,
        "fsize_title_pt": 20,
        "fsize_plot_title_pt": 24,
        "title_margin": 50,
        "markersize": markersize,
        "ticklen": 15,
        "tickwidth": 5,
        "axislinewidth": 3,
        "markerlinewidth": 0.5,  # 0.9,
        "cbar_thickness": 20,
        "cbar_len": 0.75,
    }

    # If style is provided, update default_style with user values
    if style is not None:
        default_style.update(style)
    style = default_style

    if sizing is not None:
        default_sizing.update(sizing)
    sizing = default_sizing

    if sizing["fig_width"] is None:
        # Size the figure to the lattice, so a marker of `markersize` px tiles
        # the grid whatever the lattice covers. The +2 is one marker of padding
        # at each end; the +54 mm is the room plotly's colorbar takes. This
        # reproduces the 206/260 mm defaults for a single medulla lattice.
        x_span = (background_hex["x"].max() - background_hex["x"].min()) + 2
        y_span = (
            (background_hex["y"].max() - background_hex["y"].min()) + 2
        ) / _HEX_ASPECT
        sizing["fig_width"] = round(
            sizing["fig_height"] * x_span / y_span + (54 if colorbar else 0)
        )

    # Constants for unit conversion
    POINTS_PER_INCH = 72  # Typography standard: 1 point = 1/72 inch
    MM_PER_INCH = 25.4  # Standard conversion: 1 inch = 25.4 mm

    # sizing of the figure and font
    pixelsperinch = dpi  # Use the provided DPI value
    pixelspermm = pixelsperinch / MM_PER_INCH

    # Default colorscale
    if custom_colorscale is None:
        custom_colorscale = [[0, "rgb(255, 255, 255)"], [1, "rgb(0, 20, 200)"]]

    area_width = (sizing["fig_width"] - sizing["fig_margin"]) * pixelspermm
    area_height = (sizing["fig_height"] - sizing["fig_margin"]) * pixelspermm

    fsize_ticks_px = sizing["fsize_ticks_pt"] * (1 / POINTS_PER_INCH) * pixelsperinch
    fsize_title_px = sizing["fsize_title_pt"] * (1 / POINTS_PER_INCH) * pixelsperinch
    fsize_plot_title_px = (
        sizing["fsize_plot_title_pt"] * (1 / POINTS_PER_INCH) * pixelsperinch
    )

    # Get global min and max for consistent color scale
    # minimum of 0 and df.values.min()
    vals = df.to_numpy()
    if global_min is None:
        global_min = min(0, vals.min())
    if global_max is None:
        global_max = vals.max()

    # Symbol number to choose to plot hexagons
    symbol_number = 15

    # initiate plot
    fig = go.Figure()
    top_margin = sizing["title_margin"] if title else 0
    fig.update_layout(
        autosize=False,
        height=area_height,
        width=area_width,
        margin={"l": 0, "r": 0, "b": 0, "t": top_margin, "pad": 0},
        paper_bgcolor=style["papercolor"],
        plot_bgcolor=style["papercolor"],
        title=(
            dict(
                text=title,
                x=0.5,
                xanchor="center",
                font=dict(size=fsize_plot_title_px, family=style["font_type"]),
            )
            if title
            else None
        ),
    )
    # scaleanchor/scaleratio is what keeps the hexagons regular; see
    # `equal_aspect` in the docstring.
    hex_aspect = (
        {"scaleanchor": "y", "scaleratio": _HEX_ASPECT} if equal_aspect else {}
    )
    fig.update_xaxes(
        showgrid=False,
        showticklabels=False,
        showline=False,
        visible=False,
        **hex_aspect,
    )
    fig.update_yaxes(
        showgrid=False, showticklabels=False, showline=False, visible=False
    )

    # Convert index values (formatted as '-12,34') into separate x and y coordinates
    df = df[(df.index != "nan") & (~df.index.isnull())]
    coords = [tuple(map(float, idx.split(","))) for idx in df.index]
    x_vals, y_vals = zip(*coords)  # Separate into x and y lists

    # Everything is drawn; columns off the lattice just get no background
    # hexagon, so flag them rather than silently accepting a frame mismatch.
    _warn_unplaced(x_vals, y_vals, background_hex, which_eye)

    if isinstance(df, pd.Series) or len(df.columns) == 1:
        if isinstance(df, pd.DataFrame):
            df = df.iloc[:, 0]
        fig.add_trace(bg_hex())
        fig.add_trace(data_hex(df))

    elif isinstance(df, pd.DataFrame):
        # Adjust figure size - add extra height for slider
        slider_height = 100  # pixels
        area_height += slider_height

        # Create frames for slider
        frames = []
        slider_steps = []

        # Add base layout
        top_margin = sizing["title_margin"] if title else 0
        fig.update_layout(
            autosize=False,
            height=area_height,
            width=area_width,
            margin={
                "l": 0,
                "r": 0,
                "b": slider_height,
                "t": top_margin,
                "pad": 0,
            },  # Add bottom margin for slider
            paper_bgcolor=style["papercolor"],
            plot_bgcolor=style["papercolor"],
            title=(
                dict(
                    text=title,
                    x=0.5,
                    xanchor="center",
                    font=dict(size=fsize_plot_title_px, family=style["font_type"]),
                )
                if title
                else None
            ),
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {
                        "font": {"size": 16},
                        "visible": True,
                        "xanchor": "right",
                    },
                    "pad": {"b": 10, "t": 0},  # Adjusted padding
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,  # Move slider below plot
                    "steps": [],
                }
            ],
        )

        # Create frames for each column
        for i, col_name in enumerate(df.columns):
            series = df[col_name]
            frame_data = [
                bg_hex(),
                data_hex(series),
            ]

            frames.append(go.Frame(data=frame_data, name=str(i)))

            # Add to slider
            slider_steps.append(
                {
                    "args": [
                        [str(i)],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                    ],
                    "label": col_name,
                    "method": "animate",
                }
            )

            # Set initial display to first column
            if i == 0:
                fig.add_traces(frame_data)

        # Update slider with all steps
        fig.layout.sliders[0].steps = slider_steps
        fig.frames = frames

        # Update axes
        fig.update_xaxes(
            showgrid=False, showticklabels=False, showline=False, visible=False
        )
        fig.update_yaxes(
            showgrid=False, showticklabels=False, showline=False, visible=False
        )

    else:
        # raise error
        raise ValueError("df must be a pd.Series or pd.DataFrame")

    return fig


def looming_stimulus(start_coords, all_coords, n_time=4):
    """
    Generate a list of lists of coordinates for a looming stimulus. The stimulus starts
    at the start_coords and expands outwards in a hexagonal pattern. The stimulus
    expands for n_time steps. Currently the expansion happens one layer at a time.

    Args:
        start_coords (list): List of strings of the form 'x,y' where x and y are the
            coordinates of the starting hexes for the stimulus.
        all_coords (list): List of strings of the form 'x,y' where x and y are the
            coordinates of all hexes in the grid.
        n_time (int): Default=4. Number of time steps for the stimulus to expand.

    Returns:
        stim_str (list):
            List of lists of strings of the form 'x,y' where x and y are the coordinates
            of the hexes that are stimulated at each time step.
    """
    coords = [tuple(map(float, idx.split(","))) for idx in all_coords]
    x_vals, y_vals = zip(*coords)  # Separate into x and y lists

    # sort and rank x_vals
    x_sorted = sorted(list(set(x_vals)))
    x_to_rank = {x: rank for rank, x in enumerate(x_sorted)}
    rank_to_x = {rank: x for rank, x in enumerate(x_sorted)}
    y_sorted = sorted(list(set(y_vals)))
    y_to_rank = {y: rank for rank, y in enumerate(y_sorted)}
    rank_to_y = {rank: y for rank, y in enumerate(y_sorted)}

    start = [tuple(map(float, idx.split(","))) for idx in start_coords]
    stimulus = []
    stimulus.append(start)
    for atime in range(n_time):
        for x, y in start:
            start_copy = start.copy()
            # hexes above and below x
            if y_to_rank[y] + 2 in rank_to_y:
                start_copy.append((x, rank_to_y[y_to_rank[y] + 2]))
            if y_to_rank[y] - 2 in rank_to_y:
                start_copy.append((x, rank_to_y[y_to_rank[y] - 2]))
            # hexes to the left
            if x_to_rank[x] + 1 in rank_to_x:
                if y_to_rank[y] + 1 in rank_to_y:
                    start_copy.append(
                        (rank_to_x[x_to_rank[x] + 1], rank_to_y[y_to_rank[y] + 1])
                    )
                if y_to_rank[y] - 1 in rank_to_y:
                    start_copy.append(
                        (rank_to_x[x_to_rank[x] + 1], rank_to_y[y_to_rank[y] - 1])
                    )
            # hexes to the right
            if x_to_rank[x] - 1 in rank_to_x:
                if y_to_rank[y] + 1 in rank_to_y:
                    start_copy.append(
                        (rank_to_x[x_to_rank[x] - 1], rank_to_y[y_to_rank[y] + 1])
                    )
                if y_to_rank[y] - 1 in rank_to_y:
                    start_copy.append(
                        (rank_to_x[x_to_rank[x] - 1], rank_to_y[y_to_rank[y] - 1])
                    )

            start = list(set(start_copy))
        stimulus.append(start)

    stim_str = []
    for atime in range(n_time):
        stim_atime = []
        for x, y in stimulus[atime]:
            # Format x and y to remove .0 if they're integers
            x_str = str(int(x)) if x == int(x) else str(x)
            y_str = str(int(y)) if y == int(y) else str(y)
            stim_atime.append(f"{x_str},{y_str}")
        stim_str.append(stim_atime)
    return stim_str


def make_sine_stim(phase=0, amplitude=1, n=8):
    """
    Generate a dictionary of values representing a sine wave stimulus with a given phase
    and amplitude. The sine wave is defined over n points, starting from the given phase.

    Args:
        phase (int): Phase of the sine wave in degrees. Default is 0.
        amplitude (float): Amplitude of the sine wave. Default is 1.
        n (int): Number of points in the sine wave. Default is 8.

    Returns:
        dict:
            A dictionary where keys are indices from 1 to n, and values are the
            corresponding sine wave values.
    """
    x = (phase % 180) / 180 * np.pi
    x = np.linspace(x, x + np.pi, n)
    y = amplitude * abs(np.sin(x))
    return dict(zip(range(1, n + 1), y))


def plot_mollweide_projection(
    data: pd.Series | pd.DataFrame,
    fig_size: tuple = (900, 700),
    custom_colorscale: Optional[Union[list, str]] = None,
    global_min: Optional[float] = None,
    global_max: Optional[float] = None,
    dataset: str = "Zhao2024",
    eyemap_dir: Optional[str] = None,
    which_eye: str = "right",
    marker_size: int = 8,
    value_name: str = "weight",
    colorbar: bool = True,
) -> go.Figure:
    """
    Generates a heatmap to visualize the value of column features per column using the
    mollweide projection.

    Args:
        data (pd.Series | pd.DataFrame): Data with index formatted as strings of the
            form '-12,34', where the first number is the x-coordinate and the second
            number is the y-coordinate. The data to plot. Each column will generate a
            separate frame in the plot.
        fig_size (tuple): Size of the figure in pixels (width, height).
        custom_colorscale (list | str, optional): Custom colorscale for the heatmap. If
            None, defaults to white-to-blue colorscale [[0, "rgb(255, 255, 255)"],
            [1, "rgb(0, 20, 200)"]]. Could also be a string e.g. 'Viridis' or 'Reds'.
        global_min (float | None): Global minimum value for the color scale. If this
            minumum is >0, 0 is used.
        global_max (float | None): Global maximum value for the color scale. If None,
            the maximum value of the data is used.
        dataset (str): The dataset to use for the hexagon locations. Options are:

            - 'Zhao2024': mapping from hexagonal coordinates to 3D coordinates, update from Zhao et al. 2022 (https://www.biorxiv.org/content/10.1101/2022.12.14.520178v1).

        eyemap_dir (Optional[str]): Path to a **directory** holding one eyemap
            file per hemisphere — ``right.xlsx`` and ``left.xlsx``, each with
            ``hex1,hex2,x,y,z`` columns, as written by the eyemap archive's
            ``download_eyemaps.py`` (``.xls``/``.csv`` also accepted).
            ``which_eye`` picks which of them to read. When given, this takes
            precedence over ``dataset``. Any ``p``/``q`` columns are ignored;
            see ``_load_local_eyemap()``.
        which_eye (str): Default='right'. Which eye to project, one of 'right',
            'left', 'both'; see ``hex_heatmap()`` for how this chooses both the
            file(s) and the hex id columns, and hence which frame ``data``'s
            index must be in. Each row keeps its own eye's viewing direction, so
            'both' projects the two hemifields correctly; a single eye simply
            leaves the other hemifield blank. Only 'right' is valid without
            ``eyemap_dir``: the bundled dataset covers the right optic lobe only.
        marker_size (int): Size of markers in the plot.

    Returns:
        go.Figure:
            A Plotly figure object containing the mollweide projection heatmap.

    Warns:
        UserWarning: When some rows of ``data`` have no matching ``hex1``/
            ``hex2`` in the eyemap and so cannot be placed on the sphere. Those
            rows are dropped from the figure. Rim columns present in the
            connectome but absent from the eyemap are the usual cause.
    """

    def cart2sph(xyz: np.array) -> np.array:
        """
        Convert Cartesian to spherical coordinates.
        Theta is polar angle (from +z), phi is angle from +x to +y.
        """
        r = np.sqrt((xyz**2).sum(1))
        theta = np.arccos(xyz[:, 2])
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        phi[phi < 0] = phi[phi < 0] + 2 * np.pi
        return np.stack((r, theta, phi), axis=1)

    def sph2Mollweide(thetaphi: np.array) -> np.array:
        """
        Spherical (viewed from outside) to Mollweide,
            cf. https://mathworld.wolfram.com/MollweideProjection.html
        """
        azim = thetaphi[:, 1]
        azim[azim > np.pi] = azim[azim > np.pi] - 2 * np.pi  # longitude/azimuth
        elev = np.pi / 2 - thetaphi[:, 0]  # lattitude/elevation in radian

        N = len(azim)  # number of points
        xy = np.zeros((N, 2))  # output
        for i in range(N):
            theta = np.arcsin(2 * elev[i] / np.pi)
            if np.abs(np.abs(theta) - np.pi / 2) < 0.001:
                xy[i,] = [
                    2 * np.sqrt(2) / np.pi * azim[i] * np.cos(theta),
                    np.sqrt(2) * np.sin(theta),
                ]
            else:
                # to calculate theta
                dtheta = 1
                while dtheta > 1e-3:
                    theta_new = theta - (
                        2 * theta + np.sin(2 * theta) - np.pi * np.sin(elev[i])
                    ) / (2 + 2 * np.cos(2 * theta))
                    dtheta = np.abs(theta_new - theta)
                    theta = theta_new
                xy[i,] = [
                    2 * np.sqrt(2) / np.pi * azim[i] * np.cos(theta),
                    np.sqrt(2) * np.sin(theta),
                ]
        return xy

    def create_mollweide_guidelines():
        """
        Create Mollweide projection guidelines as plotly traces
        """
        traces = []

        # Create meridians
        ww = np.stack((np.linspace(0, 180, 19), np.repeat(-180, 19)), axis=1)
        w = np.stack((np.linspace(180, 0, 19), np.repeat(-90, 19)), axis=1)
        m = np.stack((np.linspace(0, 180, 19), np.repeat(0, 19)), axis=1)
        e = np.stack((np.linspace(180, 0, 19), np.repeat(90, 19)), axis=1)
        ee = np.stack((np.linspace(0, 180, 19), np.repeat(180, 19)), axis=1)
        pts = np.vstack((ww, w, m, e, ee))
        rtp = np.insert(pts / 180 * np.pi, 0, np.repeat(1, pts.shape[0]), axis=1)
        meridians_xy = sph2Mollweide(rtp[:, 1:3])

        traces.append(
            go.Scatter(
                x=meridians_xy[:, 0],
                y=meridians_xy[:, 1],
                mode="lines",
                line=dict(color="lightgrey", width=0.5),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Create parallels
        for lat in [45, 90, 135]:
            pts = np.stack((np.repeat(lat, 37), np.linspace(-180, 180, 37)), axis=1)
            rtp = np.insert(pts / 180 * np.pi, 0, np.repeat(1, pts.shape[0]), axis=1)
            parallel_xy = sph2Mollweide(rtp[:, 1:3])

            traces.append(
                go.Scatter(
                    x=parallel_xy[:, 0],
                    y=parallel_xy[:, 1],
                    mode="lines",
                    line=dict(color="lightgrey", width=0.5),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        return traces

    def create_data_scatter(series_data, x_coords, y_coords, column_name=None):
        """Create scatter plot for data points"""
        return go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers",
            marker=dict(
                color=series_data.values,
                colorscale=custom_colorscale,
                cmin=global_min,
                cmax=global_max,
                size=marker_size,
                colorbar=(
                    None
                    if not colorbar
                    else dict(
                        title=dict(text=value_name, side="right"),
                    )
                ),
            ),
            customdata=np.stack([x_coords, y_coords, series_data.values], axis=-1),
            hovertemplate="x: %{customdata[0]:.2f}<br>y: %{customdata[1]:.2f}<br>%{text}: %{customdata[2]:.4f}<extra></extra>",
            text=[value_name] * len(series_data),
            showlegend=False,
        )

    # Default colorscale
    if custom_colorscale is None:
        custom_colorscale = [[0, "rgb(255, 255, 255)"], [1, "rgb(0, 20, 200)"]]

    # Clean data - remove NaN indices
    data = data[(data.index != "nan") & (~data.index.isnull())]

    # Load eyemap data and convert coordinates
    hex1_col, hex2_col = _hex_columns(which_eye)
    if eyemap_dir is not None:
        ucl_hex = _load_local_eyemap(eyemap_dir, which_eye, extra_cols=("x", "y", "z"))
    elif which_eye != "right":
        raise ValueError(
            f"which_eye={which_eye!r} needs eyemap_dir: the bundled dataset "
            f"({dataset!r}) covers the right optic lobe only."
        )
    else:
        ucl_hex = load_dataset(dataset)

    # Convert string indices to coordinate arrays
    coords = [tuple(map(float, idx.split(","))) for idx in data.index]
    coord_array = np.array(coords)

    # Get global min and max for consistent color scale
    vals = data.to_numpy() if isinstance(data, pd.DataFrame) else data.values
    if global_min is None:
        global_min = min(0, vals.min())
    if global_max is None:
        global_max = vals.max()

    rtp2 = cart2sph(ucl_hex[["x", "y", "z"]].values)
    xy = sph2Mollweide(rtp2[:, 1:3])
    xy[:, 0] = -xy[:, 0]  # flip x axis
    # Internal names stay hex2/hex1 whichever frame the eyemap columns are in.
    hex_moll = np.concatenate((xy, ucl_hex[[hex2_col, hex1_col]].values), axis=1)
    hex_moll = pd.DataFrame(hex_moll, columns=["x", "y", "hex2", "hex1"])
    hex_moll[["hex2", "hex1"]] = hex_moll[["hex2", "hex1"]].astype(int)

    # Convert data coordinates to Mollweide
    hex1_id = (coord_array[:, 1] - coord_array[:, 0]) / 2
    hex2_id = (coord_array[:, 1] + coord_array[:, 0]) / 2

    coord_df = pd.DataFrame({"hex1_id": hex1_id, "hex2_id": hex2_id}, index=data.index)

    merged_coords = coord_df.merge(
        hex_moll, left_on=["hex1_id", "hex2_id"], right_on=["hex1", "hex2"], how="left"
    )

    x_mollweide = merged_coords["x"].values
    y_mollweide = merged_coords["y"].values

    # Rows with no matching lattice hex have no direction on the sphere, so they
    # drop out of the figure. Say so rather than losing them silently.
    n_unplaced = int(np.isnan(x_mollweide).sum())
    if n_unplaced:
        warnings.warn(
            f"{n_unplaced} of {len(x_mollweide)} rows have no matching "
            f"{hex1_col}/{hex2_col} in the eyemap (which_eye={which_eye!r}) and "
            "are left out of the projection. Columns present in the connectome "
            "but absent from the eyemap -- typically rim columns with no fitted "
            "viewing direction -- are the usual cause; if nearly every row is "
            "unplaced, the data's index is probably in the other hex frame.",
            UserWarning,
            stacklevel=2,
        )

    # Create figure
    fig = go.Figure()

    # Add guidelines
    guidelines = create_mollweide_guidelines()
    for trace in guidelines:
        fig.add_trace(trace)

    # Handle single series vs DataFrame
    if isinstance(data, pd.Series) or (
        isinstance(data, pd.DataFrame) and len(data.columns) == 1
    ):
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]

        # Single plot
        fig.add_trace(create_data_scatter(data, x_mollweide, y_mollweide))

    elif isinstance(data, pd.DataFrame):
        # Multiple columns - create frames for slider
        frames = []
        slider_steps = []

        # Create frames for each column
        for i, col_name in enumerate(data.columns):
            series = data[col_name]

            # Create frame data (guidelines + data scatter)
            frame_traces = guidelines + [
                create_data_scatter(series, x_mollweide, y_mollweide)
            ]
            frames.append(go.Frame(data=frame_traces, name=str(i)))

            # Add slider step
            slider_steps.append(
                {
                    "args": [
                        [str(i)],
                        {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                    ],
                    "label": col_name,
                    "method": "animate",
                }
            )

            # Set initial display to first column
            if i == 0:
                fig.add_trace(
                    create_data_scatter(series, x_mollweide, y_mollweide, col_name)
                )

        # Add frames to figure
        fig.frames = frames

        # Add slider
        fig.update_layout(
            sliders=[
                {
                    "active": 0,
                    "currentvalue": {
                        "font": {"size": 16},
                        "visible": True,
                        "xanchor": "right",
                    },
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": slider_steps,
                }
            ]
        )

    # Update layout
    fig.update_layout(
        width=fig_size[0],
        height=fig_size[1],
        xaxis=dict(
            range=[-np.pi, np.pi],
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            showticklabels=False,
            showline=False,
            visible=False,
        ),
        yaxis=dict(
            range=[-np.pi / 2, np.pi / 2],
            showgrid=False,
            showticklabels=False,
            showline=False,
            visible=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(
            l=0,
            r=0,
            t=50,
            b=50 if isinstance(data, pd.DataFrame) and len(data.columns) > 1 else 0,
        ),
    )

    return fig
