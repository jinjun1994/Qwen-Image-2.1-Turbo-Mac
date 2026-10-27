"""Frozen worker entrypoint; all inference remains in the native MLX backend."""
import os
os.environ['USE_TORCH']='0'
os.environ['USE_TF']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
import sys
import certifi
os.environ.setdefault("SSL_CERT_FILE",certifi.where())
if '--runtime-check' in sys.argv:
    import traceback
    try:
        import mlx.core as mx
        print(mx.device_info(),flush=True)
        from mqwen.native import Qwen21Transformer
        print('native imports OK',flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    raise SystemExit(0)
from mqwen.cli import main
if __name__=='__main__':
    raise SystemExit(main())
