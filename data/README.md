# Data

| File | Source | Licence |
|---|---|---|
| `esol.csv` | ESOL (Delaney, *J. Chem. Inf. Comput. Sci.* 2004, 44, 1000) as distributed with MoleculeNet | As distributed by MoleculeNet |
| `aqsoldb.csv` | AqSolDB `results/data_curated.csv` (Sorkun, Khetan and Er, *Sci. Data* 2019, 6, 143), from https://github.com/mcsorkun/AqSolDB | MIT (repository); data released under CC0 1.0 |

`aqsoldb.csv` is the unmodified curated release; `solnoise.data.load_aqsoldb` checks its SHA-256
(363a42a4e4ea8e6039758146d34efdd95cde2da81c665316cacc1869e0a805ed) and applies the curation described in its docstring.
