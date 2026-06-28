# Launch !aBrowser in dev mode with the full native-build environment.
# Usage:  ./dev.ps1     (from the repo root, in PowerShell)
$ErrorActionPreference = "Stop"

# 1. Import the MSVC build environment (cl.exe / INCLUDE / LIB)
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

# 2. Toolchain + PATH (standalone CMake/Ninja first, then cargo)
$machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$user = [System.Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "C:\Program Files\CMake\bin;$env:Path;$machine;$user;$env:USERPROFILE\.cargo\bin"
$env:CMAKE = "C:\Program Files\CMake\bin\cmake.exe"
$env:CMAKE_GENERATOR = "Ninja"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
$env:PROTOC = "C:\Users\mrjar\tools\protoc\bin\protoc.exe"
$env:VULKAN_SDK = "C:\VulkanSDK\1.4.350.0"

# 3. Short build path (Windows MAX_PATH) — map B: -> src-tauri\target
if (-not (Test-Path "B:\")) {
  subst B: "$PSScriptRoot\src-tauri\target"
}
$env:CARGO_TARGET_DIR = "B:\"

Set-Location $PSScriptRoot
npm run tauri dev
