"""Repository-wide pytest bootstrap.

The installed Arrow/native stack exits inside ``libarrow.so`` when OpenPI/JAX is imported before
PyArrow in the same process. Importing PyArrow first is the established safe order used by the
offline diagnostics; enforce it before pytest imports any test module as well.
"""

import pyarrow.parquet as _pyarrow_parquet  # noqa: F401
