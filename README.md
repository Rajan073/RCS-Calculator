# ✈️ Ultra-Realism Aircraft RCS Analysis Suite

> **Physical Optics + Physical Theory of Diffraction + 2-Bounce Geometric Optics**
> Full-polarimetry monostatic RCS solver with ITU-R P.676-12 atmosphere, Swerling statistics, and an 8-panel 2K dashboard.

---

## 📋 Table of Contents

- [What This Does](#what-this-does)
- [Physics Engine](#physics-engine)
- [Supported File Formats](#supported-file-formats)
- [Requirements](#requirements)
- [Installation](#installation)
  - [macOS (Primary Platform)](#macos-primary-platform)
  - [Windows](#windows)
- [Configuration — Edit These Lines](#configuration--edit-these-lines)
- [Running the Suite](#running-the-suite)
- [Output Files](#output-files)
- [Dashboard Panels](#dashboard-panels)
- [Material Database](#material-database)
- [Radar Bands](#radar-bands)
- [Known Limitations](#known-limitations)
- [References](#references)

---

## What This Does

This suite computes the **monostatic Radar Cross Section (RCS)** of any 3-D CAD target (aircraft, missile, ship, satellite) across a full 360° azimuth sweep at a chosen radar band, polarisation, and elevation angle. It outputs:

- A terminal report with key RCS values
- An 8-panel 2K PNG dashboard (polar plot, Cartesian, CDF, histogram, Doppler, frequency sweep)
- Atmospheric correction using the real ITU-R P.676-12 gaseous attenuation model

---

## Physics Engine

| Module | Method | Reference |
|---|---|---|
| Surface scattering | Physical Optics (PO) with Fresnel reflection coefficients | Knott et al. [1], Balanis [3] |
| Edge diffraction | Physical Theory of Diffraction — Ufimtsev (1962) fringe waves | Ufimtsev [2] |
| Corner reflectors | 2-Bounce Geometric Optics (GO) dihedral detection | Knott et al. [1] Ch.4 |
| Polarisation | Full HH / VV / average | Balanis [3] §5.2–5.3 |
| Atmosphere | ISA + ITU-R P.676-12 O₂ + H₂O line-by-line | ITU-R P.676-12 [5] |
| Target statistics | Swerling Models 0 / I / II / III / IV | Skolnik [4] Ch.2 |

---

## Supported File Formats

| Format | Extension | Loader | Notes |
|---|---|---|---|
| STL | `.stl` | trimesh | Fastest — recommended for iteration |
| OBJ | `.obj` | trimesh | Supports multi-part geometry |
| PLY | `.ply` | trimesh | |
| STEP | `.step` `.stp` | trimesh (OpenCASCADE) | Onshape / SolidWorks / CATIA export |
| **IGES** | `.iges` `.igs` | **gmsh OpenCASCADE kernel** | Requires gmsh — most accurate B-rep tessellation |
| Parasolid | `.x_t` `.x_b` | FreeCAD CLI → gmsh | Requires FreeCAD installed |

> **Why gmsh for IGES?**
> IGES stores NURBS B-rep geometry, not triangles. trimesh has no NURBS evaluator and silently fails or produces degenerate meshes. gmsh bundles the full OpenCASCADE geometry kernel — the same one used in FreeCAD, Salome, and Code_Aster — which heals B-rep topology gaps and runs a Frontal-Delaunay surface mesher.

---

## Requirements

### Python packages

```
numpy
trimesh
matplotlib
rich
gmsh          # for IGES / STEP B-rep tessellation
```

### System tools

| Tool | Required for | macOS install | Windows install |
|---|---|---|---|
| gmsh | IGES + STEP loading | `pip install gmsh` | `pip install gmsh` |
| FreeCAD | Parasolid `.x_t` / `.x_b` only | `brew install --cask freecad` | Download from freecad.org |

---

## Installation

### macOS (Primary Platform)

This suite was developed and tested on **macOS (Apple Silicon + Intel)**. All paths in the code use Unix-style forward slashes.

```bash
# 1. Clone the repo
git clone https://github.com/yourname/rcs-suite.git
cd rcs-suite

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install numpy trimesh matplotlib rich gmsh

# 4. (Optional) Install FreeCAD for Parasolid .x_t support
brew install --cask freecad
```

Verify gmsh works:
```bash
python3 -c "import gmsh; print('gmsh OK')"
```

---

### Windows

The code runs on Windows with **two path changes** and one shell difference. Everything else is identical.

#### Step 1 — Install Python dependencies

```powershell
pip install numpy trimesh matplotlib rich gmsh
```

#### Step 2 — Change the paths in the script

Open `PCS_Suite_v3.py` and find the **PATHS block** near the top (around line 70):

```python
# ── CHANGE THESE THREE LINES ─────────────────────────────────────────────────
ROOT_DIR   = Path("/Users/apple/Visual studio code/Radar Cross section analysis")
OUTPUT_DIR = ROOT_DIR / "Outputs"
CAD_FILE   = Path("/Users/apple/Downloads/Arch Angel MK3.stl")
```

Replace with your Windows paths using **raw strings** (prefix `r` before the quote) to avoid backslash issues:

```python
# ── WINDOWS PATHS ─────────────────────────────────────────────────────────────
ROOT_DIR   = Path(r"C:\Users\YourName\Documents\RCS Analysis")
OUTPUT_DIR = ROOT_DIR / "Outputs"
CAD_FILE   = Path(r"C:\Users\YourName\Downloads\Arch Angel MK3.stl")
```

> ⚠️ On Windows, `Path("C:\Users\...")` will fail silently because `\U` and `\D` are interpreted as escape sequences. Always use `r"..."` raw strings or forward slashes: `Path("C:/Users/YourName/...")`.

#### Step 3 — FreeCAD path (only if using `.x_t` / `.x_b` Parasolid files)

On macOS, `FreeCADCmd` is found automatically if installed via Homebrew.
On Windows, you must add FreeCAD to your system PATH, or hardcode the full path in the `load_and_validate_geometry` function:

```python
# Find this line in the .x_t / .x_b branch (around line 310):
cmd = ["FreeCADCmd", "-c", ...]

# Change to full path on Windows:
cmd = [r"C:\Program Files\FreeCAD 0.21\bin\FreeCADCmd.exe", "-c", ...]
```

#### Step 4 — Run

```powershell
python PCS_Suite_v3.py
```

---

## Configuration — Edit These Lines

All user-editable settings are at the **top of the file**. You should only ever need to change these:

```python
# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — edit these lines to point at your files
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path("/Users/apple/Visual studio code/Radar Cross section analysis")
OUTPUT_DIR = ROOT_DIR / "Outputs"
CAD_FILE   = Path("/Users/apple/Downloads/Arch Angel MK3.stl")   # ← change this
```

| Variable | Purpose |
|---|---|
| `ROOT_DIR` | Project root — where the `Outputs/` folder will be created |
| `OUTPUT_DIR` | Where PNG dashboards are saved (auto-created if missing) |
| `CAD_FILE` | Path to your target geometry file |

To switch targets, change only `CAD_FILE`. Everything else — unit detection, auto-orientation, mesh repair — is automatic.

---

## Running the Suite

```bash
python3 PCS_Suite_v3.py
```

You will be prompted interactively for:

| Prompt | Options | Default |
|---|---|---|
| Radar Band | `L S C X Ku Ka` | `X` |
| Polarization | `HH VV avg` | `HH` |
| Surface Material | See Material Database below | `Aluminum` |
| Swerling Model | `0 1 2 3 4` | `0` |
| Speed (m/s) | Any float | `280` |
| Altitude (m) | Any float | `10000` |
| Elevation angle (°) | Any float | `0` |
| Frequency sweep? | `y / n` | `n` |

---

## Output Files

All outputs are saved to `OUTPUT_DIR` (default: `ROOT_DIR/Outputs/`).

| File | Description |
|---|---|
| `RCS_<target>_<band>_<pol>_<material>.png` | 8-panel 2K dashboard at 180 DPI |

Example: `RCS_Arch Angel MK3_X_HH_Aluminum.png`

---

## Dashboard Panels

| Panel | Content |
|---|---|
| A (top-left, large) | 360° polar RCS signature — plasma colourmap |
| B | RCS vs azimuth (Cartesian) with 15° smoothed overlay |
| C | Nose ±30° vs broadside 60°–120° zoom |
| D | Doppler frequency shift vs azimuth |
| E | RCS vs frequency sweep 300 MHz–35 GHz (if enabled) |
| F | Cumulative Distribution Function (CDF) of all-aspect RCS |
| G | RCS histogram with mean and peak markers |
| H | Summary table — all parameters + key RCS values |

---

## Material Database

| Key | Description | Notes |
|---|---|---|
| `PEC` | Perfect Electric Conductor | Ideal metallic surface, `\|r\|=1` |
| `Aluminum` | Aluminium alloy 2024-T3 | Bare fuselage skin — effectively PEC at radar frequencies |
| `Titanium` | Ti-6Al-4V | Engine nacelles, hot structure |
| `CFRP` | Carbon-fibre reinforced polymer | Structural composite — partial transmission possible |
| `RAM_Salisbury` | Salisbury screen (λ/4 resonant absorber) | ~4% power reflection — narrow-band |
| `RAM_Jaumann` | Jaumann multi-layer wideband absorber | ~1.5% power reflection — 2–20 GHz |
| `RAM_IronBall` | Iron-ball paint (SR-71 type) | Ferrite-loaded lacquer — broad-angle |
| `RAM_Carbonyl` | Carbonyl iron + silicone rubber sheet | Flexible conformal absorber — S–Ka band |

---

## Radar Bands

| Key | Frequency | Wavelength |
|---|---|---|
| `L` | 1.5 GHz | 20.0 cm |
| `S` | 3.0 GHz | 10.0 cm |
| `C` | 6.0 GHz | 5.0 cm |
| `X` | 10.0 GHz | 3.0 cm |
| `Ku` | 15.0 GHz | 2.0 cm |
| `Ka` | 35.0 GHz | 0.86 cm |

---

## Known Limitations

- **PO only (no MoM / FEM):** Physical Optics is a high-frequency approximation. It breaks down when the target dimension is less than ~5λ (Rayleigh/resonance region). For small targets at low frequencies, results will be overestimated.
- **Monostatic only:** Bistatic RCS (separate transmitter and receiver) is not implemented.
- **2-bounce GO only:** The multi-bounce model handles dihedral corner reflectors analytically. It does not ray-trace arbitrary multi-bounce paths (no full SBR). For complex internal cavities (engine inlets), RCS will be underestimated.
- **Parasolid requires FreeCAD:** `.x_t` / `.x_b` files need FreeCAD installed and `FreeCADCmd` in PATH. Alternatively, export as IGES or STEP from your CAD tool.
- **Performance:** The 361-angle sweep on an 80,000-face mesh takes approximately 30–90 seconds on a 2019 MacBook Pro. The optional frequency sweep adds another 30–60 seconds.

---

## References

```
[1] Knott, E.F., Schaeffer, J.F., Tuley, M.T. — "Radar Cross Section", 2nd ed.,
    SciTech Publishing, 2004.

[2] Ufimtsev, P.Ya. — "Method of Edge Waves in the Physical Theory of
    Diffraction", Soviet Radio, 1962. USAF Foreign Technology Division
    translation FTD-HC-23-259-71, 1971.

[3] Balanis, C.A. — "Advanced Engineering Electromagnetics", 2nd ed.,
    Wiley, 2012.  §5.2–5.3 (Fresnel coefficients).

[4] Skolnik, M.I. — "Introduction to Radar Systems", 3rd ed.,
    McGraw-Hill, 2001.  Ch.2 (Swerling target models).

[5] ITU-R Recommendation P.676-12 — "Attenuation by atmospheric gases and
    related effects", ITU, 2019.

[6] Ruck, G.T. et al. — "Radar Cross Section Handbook", Vol. 1 & 2,
    Plenum Press, 1970.
```

---

## Project Structure

```
rcs-suite/
├── PCS_Suite_v3.py          # Main solver — edit CAD_FILE path at top
├── README.md
└── Outputs/                 # Auto-created — PNG dashboards saved here
```

---

*Developed on macOS. Tested with Python 3.11+, trimesh 4.x, gmsh 4.11+.*
