# Verifying the Windows wheels without CI

One Windows 11 ARM64 guest verifies **both** Windows wheels: the ARM64
interpreter runs `win_arm64` natively, and Windows' built-in x64 emulation runs
`win_amd64`. That emulation is the same compromise already accepted for the
Linux x86_64 wheel under qemu -- it proves the binary loads and decodes, not
that it behaves identically on real x64 silicon.

This exists because GitHub Actions is unavailable (billing block: runs are
created but `total_count: 0` jobs ever start, so no log is ever produced).
Once built, the VM is a permanent asset and does not depend on CI at all.

## Before you start

The VM wants ~5 GB for the ISO, 25-30 GB installed, and 6 GB of RAM. On a 16 GB
host `colima stop` first -- a Colima VM and a Windows VM cannot both be resident,
and nothing in this kit needs Docker.

Then fetch the parts that are not committed (~110 MB: two Python installers and
the dependency closure for both architectures):

```sh
./fetch-deps.sh
```

## 1. Get the ISO

<https://www.microsoft.com/software-download/windows11arm64> -- pick the ARM64
Disk Image (ISO). The download is session-gated, so it needs a browser; it is
not scriptable in a way that stays working.

## 2. Create the VM in UTM (4.6.4 is installed)

New VM -> **Virtualize** -> Windows -> select the ISO. Then:

| Setting | Value | Why |
|---|---|---|
| Memory | 6144 MB | leaves the host ~9 GB with colima stopped |
| CPU cores | 4 | of 8; oversubscribing starves the host |
| Disk | 40 GB | sparse, only grows as used |
| Import VHDX | off | installing from ISO |

Leave "Install drivers and SPICE tools" checked -- not needed for this script,
but it makes the VM usable afterwards.

Windows setup itself is GUI-only (OOBE cannot be automated from here). At the
network step, if it demands a Microsoft account, `Shift+F10` opens a console
where `oobe\bypassnro` restarts into a local-account path.

## 3. Run the verification

On the Mac, pointing at the directory holding this release's wheels:

```sh
tools/release/windows-verify/serve-kit.sh .release-work-v0.2.2/final
```

It stages both wheels, `smoke_inference_wheel.py`, the GGUF, the Python
installers and the dependency wheels into `stage/`, then serves them and prints
the exact commands to paste into the guest, version already filled in:

```powershell
cd $env:USERPROFILE
iwr http://<mac-ip>:8000/verify-wheels.ps1 -OutFile verify-wheels.ps1
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\verify-wheels.ps1 -KitUrl http://<mac-ip>:8000 -WheelVersion 0.2.2
```

A default Windows install refuses to run any `.ps1` (`PSSecurityException:
UnauthorizedAccess`). `-Scope Process` lifts that for this console only and
reverts when the window closes -- do not widen it to `LocalMachine`.

The script installs Python 3.12 ARM64 and x64 side by side (per-user, neither
touching PATH, so `python` stays unambiguous), makes a venv per wheel, installs
each wheel with its real dependencies, and runs the inference smoke.

The guest needs **no internet at all**, only a route to the Mac: the Python
installers and the full dependency closure are staged too, and pip installs with
`--no-index --find-links`. `fetch-deps.sh` resolves that closure with markers
evaluated for the guest, not for the Mac -- otherwise
`colorama; platform_system == "Windows"` evaluates false on macOS, drops out, and
the omission only shows up as a resolver failure inside the VM.

The installers' SHA-256 is pinned in `verify-wheels.ps1` and checked on both
sides; the MD5s were verified against python.org's published sums when first
downloaded.

## What counts as a pass

`WINDOWS_VERIFY_OK` on the last line, and a summary table with `PASS` for both
rows.

Every leg asserts the interpreter's architecture before installing, and throws
otherwise. The guard is deliberate: `win_arm64` and `win_amd64` wheels differ
only by platform tag, so running one on the wrong interpreter would otherwise
look exactly like a pass -- the Linux verification hit that trap when a cached
`linux/amd64` image served a request for an arm64 container.

It checks two things that are fixed when the interpreter is *built*:
`sysconfig.get_platform()` (which is also what pip uses to choose a wheel) and
the COFF machine type in the interpreter's own `python.exe`. It deliberately
does not trust `platform.machine()`: on Windows that reads
`PROCESSOR_ARCHITECTURE` out of the inherited environment block, so an emulated
x64 process launched from an ARM64 shell reports `ARM64`.

The smoke asserts more than "did not crash": it decodes real tokens, checks
greedy decoding reproduces across two runs, and checks a schema-constrained
reply parses.
