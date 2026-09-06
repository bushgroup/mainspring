# Packaging

- `mainspring.spec` -- PyInstaller build spec, driven by `tools/build_exe.ps1` (never invoke
  `pyinstaller` on this file directly: the script sets the dist/build paths). Builds onedir
  only -- a folder, no extraction -- per task 07's measurements. `entrypoint.py` is its entry
  script, since PyInstaller freezes a file, not the `pyproject.toml` console-script name.
- `mainspring.iss` -- the Inno Setup script for the installer, over the onedir build. Compile
  with Inno Setup 6's `ISCC.exe`: `iscc packaging\mainspring.iss`, output in `dist\installer\`.
- No application icon yet (`icon=None` in the spec, no `Icon` line in the `.iss`); one can be
  added to both when the viewer has one.

Task 07's measurements and the onefile-vs-onedir decision are in
`mainspring-lab/notes/packaging.md`.
