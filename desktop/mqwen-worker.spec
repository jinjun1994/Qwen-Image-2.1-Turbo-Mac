# Build on Apple Silicon using the project .venv. No model weights are bundled.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata
from pathlib import Path
root=Path(SPECPATH).parent
metadata=[]
for package in ['mlx','mflux','transformers','huggingface-hub','tokenizers','numpy','Pillow','safetensors','Send2Trash','tqdm','regex','packaging','pyyaml','requests']:
    metadata+=copy_metadata(package)
a=Analysis([str(root/'desktop/worker.py')],pathex=[str(root/'src')],
    binaries=[],datas=collect_data_files('mqwen')+collect_data_files('mlx')+collect_data_files('mflux')+metadata,
    hiddenimports=collect_submodules('mflux')+collect_submodules('mlx')+['mlx.core','mlx.nn','tokenizers','PIL.PngImagePlugin'],
    hookspath=[],runtime_hooks=[],excludes=['torch','torchvision','torchaudio','tensorflow','jax','cv2','matplotlib','scipy','av','IPython','pytest','tkinter'],noarchive=False)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,[],exclude_binaries=True,name='mqwen-worker',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=True,argv_emulation=False,target_arch='arm64',codesign_identity=None,entitlements_file=None)
coll=COLLECT(exe,a.binaries,a.datas,strip=False,upx=False,name='mqwen-worker')
