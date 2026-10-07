import typer

from pathlib import Path
from pyswip import Prolog


def infer_skipbonds(
    filename: str,
    read_folder: Path,
    write_folder: Path,
):
    prolog = Prolog()

    prolog.consult(str(read_folder / filename))
    prolog.consult("nmln/data_creation/skipBond.pl")

    results = [
        "skipBond(%s,%s).\n" % (t["X"], t["Y"]) for t in prolog.query("skipBond(X,Y).")
    ]

    with open(read_folder / filename) as f_read:
        for line in f_read:
            results.insert(0, line)

    with open(write_folder / filename, "w") as f_write:
        for result in results:
            f_write.write(result)


if __name__ == "__main__":
    typer.run(infer_skipbonds)
