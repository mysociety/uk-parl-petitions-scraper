from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
from data_common.charting import Chart, altair_sw_theme
from pandas.io.formats.style import Styler
from scipy.stats import chi2_contingency
from scipy.stats.contingency import margins


def residuals(observed, expected):
    return (observed - expected) / np.sqrt(expected)


class ChiAnalysis:
    """
    Process common chi analysis scores
    """

    significance = 1.96

    def __init__(self, df: pd.DataFrame):
        df = df.fillna(0)
        self.df = df
        self.chi2, self.p, self.dof, expected = chi2_contingency(df)
        self.expected = pd.DataFrame(expected, columns=df.columns, index=df.index)
        self.resid = residuals(self.df, self.expected)
        self.row_percent = self.df.div(df.sum(axis=1), axis=0)

    def interesting_columns(self) -> list:
        """
        return any columns with significant deviations
        """
        sig = self.__class__.significance
        cols = (
            self.resid.melt()
            .loc[lambda df: ~df["value"].between(-sig, sig)]
            .get(self.resid.columns.name)
            .unique()
            .tolist()
        )

        return cols

    def table(self, percent=False) -> Styler:
        """
        Returns styled table indicating 'interesting' values
        """

        def color_cells(s):
            if s > self.__class__.significance:
                return "background-color: #e6ffe6; color: black"
            elif s < -self.__class__.significance:
                return "background-color: #ffe6e6; color: black"
            else:
                return ""

        if percent:
            df = self.row_percent
        else:
            df = self.df

        style = df.style.apply(lambda x: self.resid.applymap(color_cells), axis=None)
        return style

    def col_table(self, col: str) -> pd.DataFrame:
        """
        generate table with analysis for a specific column
        """
        sig_size = self.__class__.significance
        raw_values = self.df[col]
        percent_values = self.row_percent[col]
        residuals = self.resid[col]
        expected = self.expected[col]
        sig = ~self.resid[col].between(-sig_size, sig_size)
        direction = np.select(
            [
                self.resid[col] < -sig_size,
                self.resid[col].between(-sig_size, sig_size),
                self.resid[col] > sig_size,
            ],
            ["Lower than expected", "Within expected range", "Higher than expected"],
        )

        df = pd.DataFrame(
            {
                "count": raw_values,
                "expected": expected,
                "row %": percent_values,
                "resid": residuals,
                "direction": direction,
            }
        )
        df.columns.name = f"Column: {col}"
        return df

    def col_chart(
        self,
        col: str,
        color: bool = True,
        df: Optional[pd.DataFrame] = None,
        invert_sort: bool = False,
    ) -> Chart:
        """
        generate a Chart for a column in the original contingency table
        """

        if df is None:
            df = self.col_table(col)

        sort = alt.EncodingSortField("row %", op="min", order="descending")
        sort = None
        if invert_sort:
            sort = alt.EncodingSortField(df.index.name, op="min", order="descending")

        x_encode = alt.X(
            "row %", title="", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])
        )
        y_encode = alt.Y(df.index.name, title="", sort=sort)

        domain = [
            "Lower than expected",
            "Within expected range",
            "Higher than expected",
        ]
        named_palette = ["sw_berry", "colour_blue_dark_30", "sw_blue"]

        color_scale = altair_sw_theme.color_scale(
            domain=domain, named_palette=named_palette
        )
        color_encode = alt.Color("direction", scale=color_scale)

        base_options = {"x": x_encode, "y": y_encode}
        color_options = {}
        if color:
            color_options["color"] = color_encode

        df_indexless = df.reset_index()
        df_indexless["text"] = df_indexless.apply(
            lambda x: f"{x['row %']:.0%} ({int(x['count'])})", axis="columns"
        )

        chart = Chart(df_indexless).mark_bar().encode(**base_options, **color_options)

        text = (
            Chart(df_indexless)
            .mark_text(color="black", dx=5, align="left")
            .encode(**base_options, text="text")
        )

        chart = chart + text
        return chart
