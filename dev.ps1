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

# 3. Short build path (Windows MAX_PATH) — map B: -> src-tauri\target.
# llama.cpp's ggml-vulkan build emits deeply-nested shader files that blow past
# MAX_PATH even with long-paths enabled, so we keep the target dir behind a short
# drive letter. `cargo clean` deletes/recreates the target dir, which leaves a
# stale (broken) B: mapping — so always recreate it cleanly rather than trusting
# a `Test-Path B:\` that lies when the mapping is dangling.
$target = "$PSScriptRoot\src-tauri\target"
if (-not (Test-Path $target)) { New-Item -ItemType Directory -Force -Path $target | Out-Null }
cmd /c "subst /D B:" 2>$null | Out-Null   # drop any existing/stale mapping
subst B: $target
$env:CARGO_TARGET_DIR = "B:\"

Set-Location $PSScriptRoot
npm run tauri dev
