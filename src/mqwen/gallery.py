"""Move an image and its sidecar to the OS trash as one recoverable folder."""
import os
from pathlib import Path
import tempfile


def trash_image(image,output):
    from send2trash import send2trash
    image=Path(image);output=Path(output).resolve()
    files=[image,image.with_suffix('.json')]
    if image.parent!=output or image.suffix.lower()!='.png':
        raise ValueError('Image is outside the output folder')
    for path in files:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError('Cannot delete a symlink or non-file output')
    existing=[path for path in files if path.exists()]
    if not existing:return False
    staging=Path(tempfile.mkdtemp(prefix='MQwen-deleted-',dir=output))
    moved=[]
    try:
        for path in existing:
            path.rename(staging/path.name);moved.append(path)
        # The OS moves one folder, keeping PNG and JSON together in the trash.
        send2trash(str(staging))
    except Exception:
        # Restore on failure without overwriting files created by another process.
        for path in moved:
            source=staging/path.name
            if source.exists():
                os.link(source,path);source.unlink()
        staging.rmdir()
        raise
    return True
