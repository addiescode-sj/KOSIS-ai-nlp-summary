"""Re-export shim (T1-28 S3a): Grid and grid moved to kosis_chat/chart/grid.py, table_part to
kosis_chat/render/table.py. Every name the old module had is re-exported by name, the same objects."""

from dataclasses import dataclass  # noqa: F401

from kosis_chat.chart.grid import Grid, grid  # noqa: F401
from kosis_chat.compute import Computed  # noqa: F401
from kosis_chat.render.table import table_part  # noqa: F401
from kosis_chat.shared.labels import item_label  # noqa: F401
from kosis_chat.shared.periods import period_text  # noqa: F401

from kosis_ai.schemas import Fact, TablePart, TableRow  # noqa: F401  # isort: skip
