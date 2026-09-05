; Inno Setup script for the mainspring installer.
;
; Packages the onedir build (dist/mainspring/, from `tools/build_exe.ps1 -Mode onedir`) --
; chosen over onefile because onefile's every-launch extraction measured 90-190+ s on this
; workstation against onedir's 2.3 s cold / 1.5 s warm (lab record, task 07). Compile with
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

[Setup]
AppId={{89C6C6F0-7686-4511-A8FF-42FA734D3B05}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
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

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\mainspring\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
