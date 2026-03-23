"""
Grid Utility Functions
======================
Provides conversions between 2D grid cell coordinates and
a single linear integer ID using the professor's formula:

    id = col + k * (row - 1)

where:
    col, row  — 1-based column and row indices
    k         — number of cells per row  (= frame_width // cell_size)

Internally the codebase uses 0-based (col, row) tuples from pixel division,
so both functions handle that conversion transparently.
"""


def cell_to_id(col_0: int, row_0: int, k: int) -> int:
    """
    Convert 0-based (col, row) grid coordinates to a single linear integer ID.

    Uses the formula: id = col + k * (row - 1)  [1-based version]
    Adapted for 0-based input:  id = (col_0 + 1) + k * row_0

    Args:
        col_0: 0-based column index  (= pixel_x // cell_size)
        row_0: 0-based row index     (= pixel_y // cell_size)
        k:     number of cells per row (frame_width // cell_size)

    Returns:
        Linear cell ID (integer, starts at 1)

    Example:
        cell_size=30, frame_width=1920 → k=64
        cell (col=0, row=0) → id = 1
        cell (col=3, row=2) → id = 4 + 64*2 = 132  [wait, let's see: (3+1) + 64*2 = 4+128 = 132]
    """
    col_1 = col_0 + 1   # convert to 1-based
    row_1 = row_0 + 1   # convert to 1-based
    return col_1 + k * (row_1 - 1)


def id_to_cell(cell_id: int, k: int) -> tuple:
    """
    Convert a linear cell ID back to 0-based (col, row) tuple.

    Inverse of cell_to_id():
        row_1 = ceil(id / k)  = (id - 1) // k + 1
        col_1 = id - k * (row_1 - 1)

    Args:
        cell_id: Linear integer ID (as returned by cell_to_id)
        k:       Number of cells per row

    Returns:
        (col_0, row_0) — 0-based grid coordinate tuple
    """
    row_1 = (cell_id - 1) // k + 1
    col_1 = cell_id - k * (row_1 - 1)
    return (col_1 - 1, row_1 - 1)   # back to 0-based
