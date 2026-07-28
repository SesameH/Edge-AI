# Copyright (c) 2026 Peter Huang.
# SPDX-License-Identifier: BSD-3-Clause

<#
.SYNOPSIS
    Run the wheel inference smoke for both Windows wheels inside a Windows 11
    ARM64 VM.

.DESCRIPTION
    Windows 11 on ARM emulates x64, so one ARM64 guest verifies both wheels:
    the ARM64 interpreter runs win_arm64, the x64 interpreter runs win_amd64
    under emulation.

    Every step asserts the interpreter's real architecture before installing.
    A wheel silently running on the wrong interpreter is the exact failure this
    script exists to catch, and pip's platform-tag check is not enough on its
    own -- py3-none-win_arm64 and py3-none-win_amd64 differ only by tag, and a
    mismatched pair would otherwise look like a pass.

.PARAMETER KitUrl
    Base URL of the staging directory served from the Mac, e.g.
    http://192.168.1.20:8000 -- see serve-kit.sh.

.PARAMETER WheelVersion
    Release being verified, e.g. 0.2.2. serve-kit.sh prints the whole command
    with this already filled in.

.EXAMPLE
    .\verify-wheels.ps1 -KitUrl http://192.168.1.20:8000 -WheelVersion 0.2.2
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$KitUrl,
    [Parameter(Mandatory = $true)][string]$WheelVersion,
    [string]$WorkDir = "$env:USERPROFILE\unirt-verify"
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'   # Invoke-WebRequest is glacial without this

$PythonVersion = '3.12.8'
# SHA-256 of the installers staged by serve-kit.sh. Their MD5s were checked
# against the ones published on python.org's release page before staging, so a
# mismatch here means the copy in the kit is not what python.org shipped.
$Targets = @(
    @{ Name = 'win_arm64'; Platform = 'win-arm64'; Machine = '0xaa64'
       Installer = "python-$PythonVersion-arm64.exe"
       Sha256 = '8f653dd553b0430c0a5c0b2e9701b46da187b61734066e8866b673a718a55f2c' }
    @{ Name = 'win_amd64'; Platform = 'win-amd64'; Machine = '0x8664'
       Installer = "python-$PythonVersion-amd64.exe"
       Sha256 = '71bd44e6b0e91c17558963557e4cdb80b483de9b0a0a9717f06cf896f95ab598' }
)

# Reads the COFF machine type out of the running interpreter's own image. On
# Windows `platform.machine()` is just PROCESSOR_ARCHITECTURE from the
# environment block, which a child inherits from its parent -- an x64
# interpreter launched from an ARM64 shell reports ARM64. sysconfig.get_platform()
# and the PE header are both fixed at build time, so neither can be fooled that
# way, and get_platform() is also what pip uses to pick the wheel tag.
$ArchProbe = @'
import struct, sys, sysconfig
with open(sys.executable, 'rb') as fh:
    head = fh.read(4096)
off = struct.unpack_from('<I', head, 0x3c)[0]
assert head[off:off + 4] == b'PE\0\0', 'not a PE image'
print(sysconfig.get_platform(), hex(struct.unpack_from('<H', head, off + 4)[0]))
'@

function Get-Kit([string]$Name, [string]$Destination) {
    if (Test-Path $Destination) {
        Write-Host "  have $Name"
        return
    }
    Write-Host "  fetching $Name"
    Invoke-WebRequest -Uri "$KitUrl/$Name" -OutFile $Destination
}

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
Set-Location $WorkDir

Write-Host "== staging from $KitUrl"
Get-Kit 'smoke_inference_wheel.py' "$WorkDir\smoke_inference_wheel.py"
Get-Kit 'model.gguf'               "$WorkDir\model.gguf"
foreach ($t in $Targets) {
    $whl = "unirt-$WheelVersion-py3-none-$($t.Name).whl"
    Get-Kit $whl "$WorkDir\$whl"
}

$results = @()

foreach ($t in $Targets) {
    $name = $t.Name
    Write-Host ""
    Write-Host "== $name"

    $root = "$WorkDir\py-$name"
    if (-not (Test-Path "$root\python.exe")) {
        $installer = "$WorkDir\$($t.Installer)"
        if (-not (Test-Path $installer)) {
            # Prefer the kit: the guest has no working network until the SPICE
            # drivers are in, and it can reach the Mac before it can reach
            # python.org.
            try {
                Write-Host "  fetching $($t.Installer) from kit"
                Invoke-WebRequest -Uri "$KitUrl/$($t.Installer)" -OutFile $installer
            } catch {
                Write-Host "  kit copy unavailable, falling back to python.org"
                $url = "https://www.python.org/ftp/python/$PythonVersion/$($t.Installer)"
                Invoke-WebRequest -Uri $url -OutFile $installer
            }
        }
        $hash = (Get-FileHash -Algorithm SHA256 $installer).Hash.ToLower()
        if ($hash -ne $t.Sha256) {
            throw "$($t.Installer) hashes $hash, expected $($t.Sha256)"
        }
        Write-Host "  installing to $root"
        # Per-user, no PATH changes: the two interpreters must not fight over
        # which one `python` means.
        Start-Process -FilePath $installer -Wait -ArgumentList @(
            '/quiet', 'InstallAllUsers=0', 'PrependPath=0', 'Include_launcher=0',
            "TargetDir=$root"
        )
    }

    $python = "$root\python.exe"
    $expected = "$($t.Platform) $($t.Machine)"
    $actual = (& $python -c $ArchProbe)
    if ($actual -ne $expected) {
        throw "interpreter at $python is [$actual], expected [$expected] -- refusing to certify $name"
    }
    Write-Host "  interpreter: $actual (as expected)"

    $venv = "$WorkDir\venv-$name"
    if (-not (Test-Path $venv)) { & $python -m venv $venv }
    $venvPython = "$venv\Scripts\python.exe"

    # No --no-deps anywhere below: `import unirt` reaches tokenizers, so a
    # partial install fails before decoding and proves nothing.
    $wheel = "$WorkDir\unirt-$WheelVersion-py3-none-$name.whl"
    $deps = "$WorkDir\deps-$name"
    $manifest = $null
    try {
        $manifest = (Invoke-WebRequest -Uri "$KitUrl/deps/$name/MANIFEST.txt").Content
    } catch {
        Write-Host "  no dependency kit for $name, pip will use PyPI"
    }

    if ($manifest) {
        New-Item -ItemType Directory -Force -Path $deps | Out-Null
        foreach ($f in $manifest -split "`r?`n" | Where-Object { $_ -ne '' }) {
            Get-Kit "deps/$name/$f" "$deps\$f"
        }
        & $venvPython -m pip install --quiet --no-index --find-links $deps $wheel
    } else {
        & $venvPython -m pip install --quiet $wheel
    }
    if ($LASTEXITCODE -ne 0) { throw "pip install failed for $name (exit $LASTEXITCODE)" }

    $loaded = (& $venvPython -c $ArchProbe)
    if ($loaded -ne $expected) {
        throw "venv interpreter is [$loaded], expected [$expected]"
    }

    Write-Host "  running inference smoke"
    & $venvPython "$WorkDir\smoke_inference_wheel.py" "$WorkDir\model.gguf"
    if ($LASTEXITCODE -ne 0) {
        throw "smoke_inference_wheel.py failed for $name (exit $LASTEXITCODE)"
    }

    $results += [pscustomobject]@{ Wheel = $name; Interpreter = $actual; Result = 'PASS' }
}

Write-Host ""
Write-Host "== summary"
$results | Format-Table -AutoSize
Write-Host "WINDOWS_VERIFY_OK"
