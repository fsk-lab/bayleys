from pathlib import Path
from itertools import cycle

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


class PlottingConfig:
    """
    Configuration class for plotting settings in matplotlib.
    """

    def __init__(
            self,
            drawing_height: float,
            drawing_base_width: float,
            left_margin: float,
            right_margin: float,
            bottom_margin: float,
            top_margin: float,
            horizontal_spacing: float,
            vertical_spacing: float,
            main_color: str,
            accent_1: str,
            color_palette: dict[str, str],
            font: str,
            font_size_major: int,
            font_size_minor: int,
            line_width_major: float = 1.5,
            line_width_minor: float = 0.5,
            dpi: int = 300,
    ):

        """
        Initializes the PlottingConfig with specified settings.

        Args:
            drawing_height (float): Height of the drawing area in inches.
            drawing_base_width (float): Base width of the drawing area in inches.
            left_margin (float): Left margin in inches.
            right_margin (float): Right margin in inches.
            bottom_margin (float): Bottom margin in inches.
            top_margin (float): Top margin in inches.
            main_color (str): "Black" color hex code for plots.
            accent_1 (str): Main highlight color hex code for plots.
            color_palette (dict[str, str]): Dictionary mapping color names to hex color codes.
            font (str): Font name or path to the font file.
            font_size_major (int): Font size for major text elements (axis labels).
            font_size_minor (int): Font size for minor text elements (axis ticks).
            line_width_major (float, optional): Line width for major lines. Defaults to 1.5.
            line_width_minor (float, optional): Line width for minor lines. Defaults to 0.5.
            dpi (int, optional): Dots per inch for figure resolution. Defaults to 300
        """
        self.drawing_height = drawing_height
        self.drawing_base_width = drawing_base_width

        self.margins = {
            "left": left_margin,
            "right": right_margin,
            "bottom": bottom_margin,
            "top": top_margin,
            "horizontal_spacing": horizontal_spacing,
            "vertical_spacing": vertical_spacing,
        }

        if Path(font).is_file():
            if not Path(font).suffix.lower() in {".ttf", ".otf"}:
                raise ValueError("The provided font file is not a valid TTF or OTF file.")
            self.font_name = Path(font).stem
            font_entry = fm.FontEntry(fname=font, name=self.font_name)
            fm.fontManager.ttflist.append(font_entry)
        else:
            self.font_name = font

        self.font = str(font)
        self.major_font_size = font_size_major
        self.minor_font_size = font_size_minor

        self.main_color = main_color
        self.accent_1 = accent_1
        self.color_palette = color_palette

        self.line_width_major = line_width_major
        self.line_width_minor = line_width_minor

        self.dpi = dpi

    def color_cycle(self, include_accent: bool = False) -> cycle:
        """
        Returns an iterator that cycles through the colors in the color palette.

        Args:
            include_accent (bool): If True, the main highlight color is also included in the color cycle.

        Returns:
            cycle: An iterator cycling through the color hex codes.
        """
        if include_accent is True:
            return cycle([self.accent_1] + list(self.color_palette.values()))
        else:
            return cycle(self.color_palette.values())

    def to_dict(self) -> dict:
        """
        Converts the PlottingConfig to a dictionary representation.

        Returns:
            dict: Dictionary containing the plotting configuration.
        """
        return {
            "drawing_height": self.drawing_height,
            "drawing_base_width": self.drawing_base_width,
            "top_margin": self.margins["top"],
            "bottom_margin": self.margins["bottom"],
            "left_margin": self.margins["left"],
            "right_margin": self.margins["right"],
            "vertical_spacing": self.margins["vertical_spacing"],
            "horizontal_spacing": self.margins["horizontal_spacing"],
            "main_color": self.main_color,
            "accent_1": self.accent_1,
            "color_palette": self.color_palette,
            "font": self.font,
            "major_font_size": self.major_font_size,
            "minor_font_size": self.minor_font_size,
            "line_width_major": self.line_width_major,
            "line_width_minor": self.line_width_minor,
            "dpi": self.dpi,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "PlottingConfig":
        """
        Creates a PlottingConfig instance from a dictionary.

        Args:
            config_dict (dict): Dictionary containing the plotting configuration.

        Returns:
            PlottingConfig: The created PlottingConfig instance.
        """
        return cls(
            drawing_height=config_dict["drawing_height"],
            drawing_base_width=config_dict["drawing_base_width"],
            left_margin=config_dict["left_margin"],
            right_margin=config_dict["right_margin"],
            bottom_margin=config_dict["bottom_margin"],
            top_margin=config_dict["top_margin"],
            horizontal_spacing=config_dict["horizontal_spacing"],
            vertical_spacing=config_dict["vertical_spacing"],
            main_color=config_dict["main_color"],
            accent_1=config_dict["accent_1"],
            color_palette=config_dict["color_palette"],
            font=config_dict["font"],
            font_size_major=config_dict["major_font_size"],
            font_size_minor=config_dict["minor_font_size"],
            line_width_major=config_dict.get("line_width_major", 1.5),
            line_width_minor=config_dict.get("line_width_minor", 0.5),
            dpi=config_dict.get("dpi", 300),
        )

    def create_figure(self, width_scaling: float, size: float = 1.0) -> tuple[plt.Figure, plt.Axes]:
        """
        Creates a matplotlib figure and axes based on the configuration settings.

        Args:
            width_scaling (float): Scaling factor for the figure width.
            size (float): Scaling factor for the figure size. Defaults to 1.0.

        Returns:
            tuple[plt.Figure, plt.Axes]: The created figure and axes.
        """
        plt.rcParams.update(
            {
                "font.size": self.major_font_size,
                "font.family": self.font_name,
                "legend.fontsize": self.minor_font_size,
                "axes.labelsize": self.major_font_size,
                "xtick.labelsize": self.minor_font_size,
                "ytick.labelsize": self.minor_font_size,
                "grid.color": "lightgray",
                "axes.linewidth": self.line_width_major,
                "xtick.major.width": self.line_width_major,
                "ytick.major.width": self.line_width_major,
                "xtick.minor.width": self.line_width_minor,
                "ytick.minor.width": self.line_width_minor,
                "grid.linewidth": self.line_width_minor,
            }
        )

        total_width = self.drawing_base_width * width_scaling * size + self.margins["left"] + self.margins["right"]
        total_height = self.drawing_height * size + self.margins["top"] + self.margins["bottom"]

        fig, ax = plt.subplots(figsize=(total_width, total_height))

        plt.subplots_adjust(
            left=self.margins["left"] / total_width,
            right=1 - self.margins["right"] / total_width,
            bottom=self.margins["bottom"] / total_height,
            top=1 - self.margins["top"] / total_height,
        )

        ax.xaxis.labelpad = self.major_font_size
        ax.yaxis.labelpad = self.major_font_size

        return fig, ax

    def create_multi_panel_figure(
            self,
            width_scaling: float,
            size: float = 1.0,
            nrows: int = 1,
            ncols: int = 1,
            sharex: bool = False,
            sharey: bool = False
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Creates a multi-panel matplotlib figure and axes based on the configuration settings.

        Args:
            width_scaling (float): Scaling factor for the width of an individual panel.
            size (float): Scaling factor for the figure size. Defaults to 1.0.
            nrows (int): Number of rows in the subplot grid. Defaults to 1.
            ncols (int): Number of columns in the subplot grid. Defaults to 1.

        Returns:
            tuple[plt.Figure, plt.Axes]: The created multi-panel figure and axes.
        """
        plt.rcParams.update(
            {
                "font.size": self.major_font_size,
                "font.family": self.font_name,
                "legend.fontsize": self.minor_font_size,
                "axes.labelsize": self.major_font_size,
                "xtick.labelsize": self.minor_font_size,
                "ytick.labelsize": self.minor_font_size,
                "grid.color": "lightgray",
                "axes.linewidth": self.line_width_major,
                "xtick.major.width": self.line_width_major,
                "ytick.major.width": self.line_width_major,
                "xtick.minor.width": self.line_width_minor,
                "ytick.minor.width": self.line_width_minor,
                "grid.linewidth": self.line_width_minor,
            }
        )

        total_width = self.drawing_base_width * width_scaling * size * ncols + self.margins["left"] + self.margins["right"] + (ncols - 1) * self.margins["horizontal_spacing"]
        total_height = self.drawing_height * size * nrows + self.margins["top"] + 2 * self.margins["bottom"] + (nrows - 1) * self.margins["vertical_spacing"]

        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(total_width, total_height),
            sharex=sharex,
            sharey=sharey
        )

        plt.subplots_adjust(
            left=self.margins["left"] / total_width,
            right=1 - self.margins["right"] / total_width,
            bottom=self.margins["bottom"] / total_height,
            top=1 - self.margins["top"] / total_height,
            wspace=self.margins["horizontal_spacing"] / (self.drawing_base_width * width_scaling * size),
            hspace=self.margins["vertical_spacing"] / (self.drawing_height * size)
        )

        return fig, axs

    def create_3d_figure(self, width_scaling: float, size: float = 1.0) -> tuple[plt.Figure, plt.Axes]:
        """
        Creates a 3D matplotlib figure and axes based on the configuration settings.

        Args:
            width_scaling (float): Scaling factor for the figure width.
            size (float): Scaling factor for the figure size. Defaults to 1.0.

        Returns:
            tuple[plt.Figure, plt.Axes]: The created 3D figure and axes.
        """
        plt.rcParams.update(
            {
                "font.size": self.major_font_size,
                "font.family": self.font_name,
                "legend.fontsize": self.minor_font_size,
                "axes.labelsize": self.major_font_size,
                "xtick.labelsize": self.minor_font_size,
                "ytick.labelsize": self.minor_font_size,
                "grid.color": "lightgray",
                "axes.linewidth": self.line_width_major,
                "xtick.major.width": self.line_width_major,
                "ytick.major.width": self.line_width_major,
                "xtick.minor.width": self.line_width_minor,
                "ytick.minor.width": self.line_width_minor,
                "grid.linewidth": self.line_width_minor,
            }
        )

        total_width = self.drawing_base_width * size * width_scaling * 2
        total_height = self.drawing_height * size * 2

        fig = plt.figure(figsize=(total_width, total_height))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([width_scaling, 1.0, 1.0])

        ax.tick_params(axis='x', pad=-self.minor_font_size)
        ax.tick_params(axis='y', pad=-self.minor_font_size)
        ax.tick_params(axis='z', pad=-self.minor_font_size)

        ax.xaxis.labelpad = self.major_font_size
        ax.yaxis.labelpad = self.major_font_size
        ax.zaxis.labelpad = self.major_font_size

        return fig, ax


