# utils/utils.py

def get_title_from_label(label: int) -> str:
    """
    Return a display title for the given label.
    Logic: "parasitized (0)" if label == 0 else "uninfected (1)".
    """
    return "parasitized (0)" if label == 0 else "uninfected (1)"
