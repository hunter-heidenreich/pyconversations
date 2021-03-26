def num2str(num):
    if num > 1_000_000_000_000:
        return f"{num / 1_000_000_000_000:.2f} T"
    elif num > 1_000_000_000:
        return f"{num / 1_000_000_000:.2f} B"
    elif num > 1_000_000:
        return f"{num / 1_000_000:.2f} M"
    elif num > 1_000:
        return f"{num / 1_000:.2f} K"
    else:
        if type(num) == int:
            return str(num)

        return str(int(num)) if num.is_integer() else f'{num:.2f}'


def time2str(ts):
    if ts > (24 * 60 * 60 * 365):
        return f'{ts / (24 * 60 * 60 * 365):.2f} yrs'
    elif ts > (24 * 60 * 60):
        return f'{ts / (24 * 60 * 60):.2f} d'
    elif ts > (60 * 60):
        return f'{ts / (60 * 60):.2f} hr'
    elif ts > 60:
        return f'{ts / 60:.2f} min'
    elif ts == 0:
        return '0 s'
    else:
        return f'{ts:.2f} s'
