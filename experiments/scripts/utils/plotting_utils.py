from matplotlib.ticker import FuncFormatter


def _thousands_formatter(x, pos):
    if x % 1000 == 0:
        return f"{int(x / 1000)}k"
    elif x > 1000:
        return f"{x / 1000:.1f}k"
    else:
        return f"{int(x)}"


def _percent_formatter(x, pos):
    return f"{int(x * 100)}%"


thousands_formatter = FuncFormatter(_thousands_formatter)
percent_formatter = FuncFormatter(_percent_formatter)
