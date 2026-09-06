# Packaging

- `mainspring.spec` -- PyInstaller build spec, driven by `tools/build_exe.ps1` (never invoke
  `pyinstaller` on this file directly: the script sets the dist/build paths and cuts PATH).
  Builds onedir only -- a folder, no extraction -- per task 07's measurements. `entrypoint.py`
  is its entry script, since PyInstaller freezes a file, not the `pyproject.toml`
  console-script name.
- `mainspring.iss` -- the Inno Setup script for the installer, over the onedir build. Compile
  with Inno Setup 6's `ISCC.exe`, which `winget install JRSoftware.InnoSetup --scope user`
  puts in `%LOCALAPPDATA%\Programs\Inno Setup 6\` and does *not* put on PATH:
  `& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" packaging\mainspring.iss`, output in
  `dist\installer\`.
- `icon/` -- the SVG artwork; `tools/make_icon.py` rasterises it into
  `src/mainspring/viewer/resources/mainspring.ico`, which the `.exe` (spec `icon=`), the
  installer (`SetupIconFile`) and the viewer's own windows all use. `check_public.py` fails
  when the `.ico` is stale against the SVGs.

Task 07's measurements and the onefile-vs-onedir decision are in
`mainspring-lab/notes/packaging.md`. The public README's "Building the executable" and
"Creating the installer" sections are the user-facing form of these steps.
