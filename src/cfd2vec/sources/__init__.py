"""Data-source plug-ins. Each module turns one dataset's native format into canonical `Case` records.

    aero_urban   AERO urban RANS corpus with coarse-twin prior
    caeml        caemldatasets-format volume files
Register a new source by adding a module with `iter_cases(...) -> Iterator[Case]`.
"""
SOURCES = ("aero_urban", "caeml")
