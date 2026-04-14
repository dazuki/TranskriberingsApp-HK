# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

fw_datas, fw_binaries, fw_hiddenimports = collect_all("faster_whisper")
ct_datas, ct_binaries, ct_hiddenimports = collect_all("ctranslate2")
sd_datas, sd_binaries, sd_hiddenimports = collect_all("sounddevice")

a = Analysis(
    ["entrypoint.py"],
    pathex=["src"],
    binaries=fw_binaries + ct_binaries + sd_binaries,
    datas=fw_datas + ct_datas + sd_datas,
    hiddenimports=fw_hiddenimports + ct_hiddenimports + sd_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="transcribe-app",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="transcribe-app",
)
