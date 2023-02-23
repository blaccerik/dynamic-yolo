def validate_page_size(page_size: str, default: int, max: int):
    if page_size is None:
        return default
    try:
        page_size = int(page_size)
    except Exception:
        page_size = default
    if page_size > max:
        page_size = max
    elif page_size <= 0:
        page_size = default
    return page_size


def validate_page_nr(page_nr: str):
    if page_nr is None:
        return 1
    try:
        page_nr = int(page_nr)
    except Exception:
        page_nr = 1
    if page_nr <= 0:
        page_nr = 1
    return page_nr