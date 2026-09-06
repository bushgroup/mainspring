; Inno Setup script for the mainspring installer.
;
; Packages the onedir build (dist/mainspring/, from `tools/build_exe.ps1`) -- onedir was
; chosen over onefile because onefile's every-launch extraction ran 90-190+ s on this
; workstation against onedir's 3-4 s to the window (lab record, task 07). Compile with
; Inno Setup 6's ISCC.exe:
;
;   iscc packaging\mainspring.iss
;
; and the installer lands in dist\installer\. Per-user install (PrivilegesRequired=lowest)
; because the viewer's settings already live in HKCU (ViewerSettings, task 06) and an
; instrument PC's operator account may not have admin rights.

#define MyAppName "mainspring"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "University of Washington"
#define MyAppURL "https://github.com/bushgroup/mainspring"
#define MyAppExeName "mainspring.exe"
; The .uimf ProgID, referenced in four places below (DefaultIcon's key, shell\open\command's
; key, .uimf's default value, and OpenWithProgids) -- one #define so a typo in one of them is
; not silent (lab record, task 15).
#define MyAppProgId "mainspring.uimf"

[Setup]
AppId={{89C6C6F0-7686-4511-A8FF-42FA734D3B05}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
; The same .ico the .exe carries (tools/make_icon.py), used for the wizard's own window
; and title bar. UninstallDisplayIcon points into the install rather than at a copy, so
; Apps & features shows the icon the installed program actually has.
SetupIconFile=..\src\mainspring\viewer\resources\mainspring.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist\installer
OutputBaseFilename=mainspring-{#MyAppVersion}-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Without this, Explorer can hold the old (or absent) .uimf association until the shell
; restarts -- ChangesAssociations makes Inno call SHChangeNotify(SHCNE_ASSOCCHANGED) after
; install and after uninstall, so the association works the moment setup finishes (lab
; record, task 15).
ChangesAssociations=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associate"; Description: "Open .uimf files with {#MyAppName}"; GroupDescription: "Other tasks:"

[Files]
Source: "..\dist\mainspring\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[Registry]
; --- Layer 1: the ProgID -- the handler's description of itself. HKA resolves to HKCU for a
; per-user install and HKLM for an elevated one, matching whichever way this install landed
; (lab record, task 15).
Root: HKA; Subkey: "Software\Classes\{#MyAppProgId}"; ValueType: string; ValueName: ""; ValueData: "mainspring UIMF file"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\{#MyAppProgId}\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"; Tasks: associate
; The inner quotes around %1 matter: without them a path containing a space arrives as two
; arguments and app.py's args[0] is a truncated path.
Root: HKA; Subkey: "Software\Classes\{#MyAppProgId}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: associate

; --- Layer 2: the extension -> ProgID link. Never touches UserChoice (see Never do this in
; the lab task) -- with no UserChoice recorded, Explorer uses these two layers directly and a
; double-click opens mainspring with no prompt; where UserChoice already points elsewhere,
; mainspring still appears in the Open-with list via OpenWithProgids.
; createvalueifdoesntexist is what keeps a *reinstall* from stealing .uimf back from whatever
; else has since claimed the default value -- install only ever fills an empty slot, never
; overwrites one. uninsdeletevalue is a separate, accepted trade-off about *uninstall*: it
; removes the value unconditionally, which is wrong only if some other program claimed .uimf
; after mainspring installed (uninstalling then clears that program's association too). The
; alternative -- leaving the value -- points .uimf at a deleted ProgID, so a .uimf opens
; nothing and explains nothing, which is the worse failure.
Root: HKA; Subkey: "Software\Classes\.uimf"; ValueType: string; ValueName: ""; ValueData: "{#MyAppProgId}"; Flags: uninsdeletevalue createvalueifdoesntexist; Tasks: associate
Root: HKA; Subkey: "Software\Classes\.uimf\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppProgId}"; ValueData: ""; Flags: uninsdeletevalue; Tasks: associate

; --- The recovery path: lets mainspring appear in Settings > Default apps as a selectable
; choice even where UserChoice already points elsewhere. Applications\mainspring.exe carries
; its own shell\open\command so Explorer's Open-with can invoke it directly, not only through
; the ProgID.
Root: HKA; Subkey: "Software\Classes\Applications\{#MyAppExeName}"; ValueType: string; ValueName: "FriendlyAppName"; ValueData: "{#MyAppName}"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Applications\{#MyAppExeName}\SupportedTypes"; ValueType: string; ValueName: ".uimf"; ValueData: ""; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Applications\{#MyAppExeName}\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Tasks: associate

; Capabilities + RegisteredApplications: the pair Settings > Default apps reads to offer
; mainspring as a default-apps choice for .uimf. HKCU RegisteredApplications has been honoured
; since Windows 8, so this works for both the per-user and the elevated install.
Root: HKA; Subkey: "Software\{#MyAppName}\Capabilities"; ValueType: string; ValueName: "ApplicationName"; ValueData: "{#MyAppName}"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\{#MyAppName}\Capabilities"; ValueType: string; ValueName: "ApplicationDescription"; ValueData: "Viewer for UIMF ion mobility mass spectrometry files"; Tasks: associate
Root: HKA; Subkey: "Software\{#MyAppName}\Capabilities\FileAssociations"; ValueType: string; ValueName: ".uimf"; ValueData: "{#MyAppProgId}"; Tasks: associate
Root: HKA; Subkey: "Software\RegisteredApplications"; ValueType: string; ValueName: "{#MyAppName}"; ValueData: "Software\{#MyAppName}\Capabilities"; Flags: uninsdeletevalue; Tasks: associate
