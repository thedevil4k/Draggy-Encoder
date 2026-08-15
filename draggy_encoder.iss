; Script de Inno Setup para Draggy Encoder.
; Se compila en CI con:  iscc /DDRAGGY_VERSION=x.y.z draggy_encoder.iss
; El directorio dist\DraggyEncoder lo produce PyInstaller (draggy_encoder.spec)
; y se copia completo, incluyendo embed-python/ (el Python embebido con pip
; bootstrapado que la app usa para instalar torch/onnxruntime a demanda).

#ifndef MyAppVersion
  #define MyAppVersion "1"
#endif

; Instalación por usuario en {localappdata}\Programs: la app escribe datos
; grandes junto al EXE (bin/ = modelos, runtimes, ffmpeg), así que no debe
; instalarse bajo Program Files (requeriría UAC para escribir).
[Setup]
AppId={{9D2A6F4B-8E11-4C5F-A3D6-1B2C4E5F6071}}
AppName=Draggy Encoder
AppVersion={#MyAppVersion}
AppVerName=Draggy Encoder v{#MyAppVersion}
AppPublisher=thedevil4k
; Icono del instalador y del desinstalador (multi-resolución, ver
; packaging/windows/make_icon.py). Sin SetupIconFile usan el icono genérico.
SetupIconFile=assets\app.ico
; Icono de la entrada en Settings > Apps / Agregar o quitar programas.
; Sin esta directiva Windows elige un icono por su cuenta (suele ser el
; genérico del desinstalador). Apuntamos al EXE, que ya lleva el .ico embebido.
UninstallDisplayIcon={app}\DraggyEncoder.exe
DefaultDirName={localappdata}\Programs\DraggyEncoder
DefaultGroupName=Draggy Encoder
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=installer
OutputBaseFilename=DraggyEncoder-Setup-{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
; El EXE, _internal/, assets/ y bin/ de arranque en limpio (sin modelos).
; bin\logs\ se excluye del copiado recursivo y se trata aparte: el sello de
; copia de Inno no debe marcar esos logs instalados (cambian en cada arranque).
Source: "dist\DraggyEncoder\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs; Excludes: "bin\logs\*"
Source: "dist\DraggyEncoder\bin\logs\*"; DestDir: "{app}\bin\logs"; Flags: skipifsourcedoesntexist uninsneveruninstall
; El Python embebido NO se marca uninsneveruninstall: va con la app.

[Icons]
Name: "{group}\Draggy Encoder"; Filename: "{app}\DraggyEncoder.exe"
Name: "{autodesktop}\Draggy Encoder"; Filename: "{app}\DraggyEncoder.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\DraggyEncoder.exe"; Description: "Launch Draggy Encoder"; Flags: nowait postinstall skipifsilent
