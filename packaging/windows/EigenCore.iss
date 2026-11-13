; EigenCore Windows installer (Inno Setup 6).
;
; Built by .github/workflows/release.yml from the one-folder app build:
;   iscc /DAppVersion=0.4.6 /DSourceDir=dist\EigenCore packaging\windows\EigenCore.iss
;
; What it does:
; - Installs for the current user only (no admin prompt) into
;   %LOCALAPPDATA%\Programs\EigenCore.
; - Adds EigenCore to the Start menu, and to Settings > Apps > Installed apps
;   (so it can be uninstalled like any other program).
; - Running a newer EigenCore-Setup.exe updates the same install in place
;   (same AppId). The in-app "Update" button runs it with /SILENT.
; - Never touches models or settings in %USERPROFILE%\.eigencore.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\EigenCore"
#endif

#define AppName "EigenCore"
#define AppExe "EigenCore.exe"
#define AppUrl "https://github.com/Khubaib7-del/EigenCore"

[Setup]
; Keep this AppId forever: it is how Windows knows a new installer is an update.
AppId={{233A1F70-891E-4BB0-B66B-6FCDD9ACEF2B}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=EigenCore
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}/issues
AppUpdatesURL={#AppUrl}/releases/latest
VersionInfoVersion={#AppVersion}.0
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\..\dist
OutputBaseFilename=EigenCore-Setup
SetupIconFile=..\..\src\eigencore\desktop\assets\eigencore.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
LicenseFile=..\..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; clear the previous version's bundled files before copying the new ones
Type: filesandordirs; Name: "{app}\_internal"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
; Interactive install: "Launch EigenCore" checkbox on the last page.
; Silent update from inside the app: runs automatically, reopening EigenCore.
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall
