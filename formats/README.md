# Adding New Format Readers

To add support for a new acquisition format, create a new Python file in this
directory and implement the `FormatReader` protocol.

## Quick Start

1. Create a new file: `formats/my_format.py`
2. Implement the reader class with `@register_format` decorator
3. Done! The format is auto-discovered on import.

## Template

```python
"""
My Format Reader

Brief description of the format.
"""
from pathlib import Path
import numpy as np

from . import register_format
from ..models import (
    Acquisition,
    AcquisitionMetadata,
    ChannelMetadata,
    OpticsMetadata,
)


@register_format
class MyFormatReader:
    """Reader for My acquisition format."""

    # Required: unique name for this format
    name = "my_format"

    # Optional: priority (higher = tried first, default 50)
    priority = 50

    @staticmethod
    def can_read(path: Path) -> bool:
        """
        Check if this reader can handle the given path.

        Should be fast - just check file existence/naming,
        don't read file contents if possible.
        """
        path = Path(path)

        # Example: check for specific file
        if (path / "metadata.json").exists():
            return True

        # Example: check for file pattern
        if path.is_dir():
            for f in path.iterdir():
                if f.name.endswith("_myformat.tif"):
                    return True

        return False

    @staticmethod
    def read(path: Path) -> Acquisition:
        """
        Read the acquisition from the path.

        Returns an Acquisition object with images and metadata.
        """
        path = Path(path)

        # 1. Read metadata (from JSON, XML, or infer from filenames)
        # ...

        # 2. Build channel metadata
        channels = [
            ChannelMetadata(name="488nm", wavelength_nm=488.0),
            ChannelMetadata(name="561nm", wavelength_nm=561.0),
        ]

        # 3. Read images
        images = {}
        for ch in channels:
            # Read your image files here
            # images[ch.name] = np.array(...)
            pass

        # 4. Build acquisition metadata
        metadata = AcquisitionMetadata(
            pixel_size_um=0.115,  # Read from metadata or use default
            channels=channels,
            z_step_um=0.5,  # None if single Z
            n_z=1,
            optics=OpticsMetadata(
                na=1.4,
                ni=1.515,
            ),
            format_name="my_format",
            source_path=path,
        )

        return Acquisition(images=images, metadata=metadata)
```

## Key Points

1. **`can_read()` should be fast** - Only check file existence and naming
   patterns. Don't parse file contents unless absolutely necessary.

2. **`name` must be unique** - Used for format identification in logs/errors.

3. **`priority` controls order** - Higher priority formats are tried first.
   Use higher priority for more specific formats (e.g., 100 for formats with
   unique identifiers, 50 for general formats, 10 for fallbacks).

4. **Images should be (Z, Y, X)** - Even for single Z-plane acquisitions,
   add a Z dimension: `image[np.newaxis, ...]`

5. **Wavelength is required** - The PSF generator needs wavelength. Either
   read from metadata or infer from channel name (e.g., "488nm" -> 488.0).

## Testing Your Format

```python
from deconv_tool.formats import load_acquisition, get_registered_formats

# Check your format is registered
print(get_registered_formats())  # Should include "my_format"

# Test loading
acq = load_acquisition("/path/to/my/acquisition")
print(acq)
```

## Common Patterns

### Reading from JSON metadata
```python
import json

metadata_path = path / "metadata.json"
with open(metadata_path) as f:
    meta = json.load(f)

pixel_size = meta.get("pixel_size_um", 0.115)
```

### Inferring wavelength from filename
```python
import re

match = re.search(r'(\d+)nm', filename)
if match:
    wavelength = float(match.group(1))
```

### Stacking multiple TIFFs
```python
import tifffile

files = sorted(path.glob("*.tif"))
stack = np.stack([tifffile.imread(f) for f in files])
```
