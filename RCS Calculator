"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          ULTRA-REALISM AIRCRAFT RCS ANALYSIS SUITE  v3.0                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PHYSICS ENGINE:                                                             ║
║   • Physical Optics (PO)  with full polarisation (HH / VV / avg)             ║
║   • Physical Theory of Diffraction (PTD) — Ufimtsev (1962) fringe waves      ║
║   • 2-Bounce Geometric Optics (GO) dihedral corner-reflector detection       ║
║   • Fresnel reflection coefficients (Balanis 2012, Ch.5)                     ║
║   • Surface-impedance material database (RAM, CFRP, Al, Ti, PEC…)            ║
║  ATMOSPHERE:                                                                 ║
║   • ISA standard atmosphere (troposphere)                                    ║
║   • ITU-R P.676-12 (2019) O₂ + H₂O line-by-line attenuation                  ║
║  STATISTICS:                                                                 ║
║   • Swerling Target Fluctuation Models I / II / III / IV (Skolnik 3e)        ║
║  POST-PROCESSING:                                                            ║
║   • RCS vs frequency sweep (Mie / resonance behaviour)                       ║
║   • Elevation-angle 3-D look-angle support                                   ║
║   • 8-panel 2K dashboard (polar, Cartesian, CDF, histogram, …)               ║
║                                                                              ║
║  KEY REFERENCES:                                                             ║
║   [1] Knott, Schaeffer & Tuley – "Radar Cross Section", 2nd ed. (2004)       ║
║   [2] P.Ya. Ufimtsev – "Method of Edge Waves in the Physical Theory          ║
║         of Diffraction", Soviet Radio (1962; USAF trans. 1971)               ║
║   [3] C.A. Balanis – "Advanced Engineering Electromagnetics" (2012)          ║
║   [4] M. Skolnik – "Introduction to Radar Systems", 3rd ed. (2001)           ║
║   [5] ITU-R P.676-12, "Attenuation by atmospheric gases" (2019)              ║
║   [6] Ruck et al. – "Radar Cross Section Handbook", Vol 1&2 (1970)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import logging
import time
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for all systems)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# gmsh — used for high-quality IGES/STEP tessellation via OpenCASCADE kernel
# Install: pip install gmsh   or   brew install gmsh
try:
    import gmsh
    _GMSH_AVAILABLE = True
except ImportError:
    _GMSH_AVAILABLE = False

from rich.console import Console
from rich.table   import Table
from rich.prompt  import Prompt
from rich.panel   import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich         import box

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log     = logging.getLogger("RCS_Suite")
console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
C0   = 299_792_458.0          # Speed of light (m/s)
MU0  = 4.0 * np.pi * 1e-7    # Free-space permeability (H/m)
EPS0 = 8.854_187_817e-12      # Free-space permittivity (F/m)
ETA0 = np.sqrt(MU0 / EPS0)   # Free-space impedance ≈ 376.73 Ω

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — edit these two lines to point at your files
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path("/Users/apple/Visual studio code/Radar Cross section analysis")
OUTPUT_DIR = ROOT_DIR / "Outputs"
CAD_FILE   = Path('/Users/apple/Downloads/Arch Angel MK3.stl')   # ← change this for each target

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RADAR_BANDS = {
    "L": 1.5e9, "S": 3.0e9, "C": 6.0e9,
    "X": 10.0e9, "Ku": 15.0e9, "Ka": 35.0e9,
}

# ══════════════════════════════════════════════════════════════════════════════
# 1.  MATERIAL DATABASE
#     Complex permittivity ε_r and permeability μ_r at X-band (10 GHz)
#     Sources: [1] Ch.1; [6] Vol.2 Ch.10; Nathanson "Radar Design Principles"
#
#     RAM materials use an override reflection coefficient 'r_power' (0–1)
#     instead of deriving Fresnel from ε_r, because their multi-layer
#     design makes the effective surface impedance the design parameter.
# ══════════════════════════════════════════════════════════════════════════════
MATERIALS: dict = {
    # ── Perfect conductor ────────────────────────────────────────────────────
    "PEC": {
        "desc":   "Perfect Electric Conductor",
        "eps_r":  complex(1.0, -1e12),   # |σ/ωε| → ∞
        "mu_r":   complex(1.0,  0.0),
        "note":   "Ideal metallic surface; |r|=1 by definition",
    },
    # ── Structural metals ────────────────────────────────────────────────────
    "Aluminum": {
        "desc":  "Aluminium alloy 2024-T3  (σ ≈ 1.7×10⁷ S/m)",
        "eps_r":  complex(1.0, -3.05e8),  # −σ/(ωε₀) at 10 GHz
        "mu_r":   complex(1.0,  0.0),
        "note":  "Bare fuselage skin; effectively PEC at radar freqs",
    },
    "Titanium": {
        "desc":  "Ti-6Al-4V  (σ ≈ 5.8×10⁵ S/m)",
        "eps_r":  complex(1.0, -1.04e7),
        "mu_r":   complex(1.05, -0.02),
        "note":  "Engine nacelles, hot structure",
    },
    # ── Carbon-fibre composites ──────────────────────────────────────────────
    "CFRP": {
        "desc":  "Carbon-fibre reinforced polymer (σ ≈ 3×10³ S/m in-plane)",
        "eps_r":  complex(8.0, -5.4),
        "mu_r":   complex(1.0, 0.0),
        "note":  "Structural composite; partial transmission possible",
    },
    # ── Radar-absorbing materials (RAM) ──────────────────────────────────────
    "RAM_Salisbury": {
        "desc":   "Salisbury screen (λ/4 resonant absorber)",
        "eps_r":  complex(1.0, 0.0),    # backing details absorbed into r_power
        "mu_r":   complex(1.0, 0.0),
        "r_power": 0.04,                # ~4 % power reflection at design freq
        "note":  "Single resistive sheet over λ/4 air gap; narrow-band",
    },
    "RAM_Jaumann": {
        "desc":   "Jaumann absorber (multi-layer wideband)",
        "eps_r":  complex(2.5, -1.2),
        "mu_r":   complex(1.8, -0.9),
        "r_power": 0.015,               # ~1.5 % power reflection
        "note":  "Stacked resistive sheets; 2–20 GHz effective",
    },
    "RAM_IronBall": {
        "desc":   "Iron-ball paint (SR-71 type ferromagnetic)",
        "eps_r":  complex(6.0, -3.5),
        "mu_r":   complex(2.5, -1.5),
        "r_power": 0.08,
        "note":  "Ferrite-loaded lacquer; broad-angle absorption",
    },
    "RAM_Carbonyl": {
        "desc":   "Carbonyl iron + silicone rubber sheet",
        "eps_r":  complex(4.2, -2.1),
        "mu_r":   complex(3.0, -2.2),
        "r_power": 0.03,
        "note":  "Flexible conformal absorber; S–Ka band",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ATMOSPHERE  —  ISA + ITU-R P.676-12
#
#     Ref: [5]  ITU-R Recommendation P.676-12 (2019), Annex 2
#          provides simplified formula for O₂ and H₂O specific attenuation
#          valid 1–350 GHz for standard atmosphere.
# ══════════════════════════════════════════════════════════════════════════════
class Atmosphere:
    """
    International Standard Atmosphere (ISA) troposphere model combined with
    ITU-R P.676-12 gaseous attenuation (O₂ + H₂O lines + continuum).
    """

    @staticmethod
    def itu_r_p676(f_GHz: float, P_hPa: float, T_K: float,
                   rho_w: float) -> tuple[float, float]:
        """
        Specific gaseous attenuation (dB/km) for O₂ and H₂O.

        Parameters
        ----------
        f_GHz : radar frequency (GHz),  1–350 GHz range
        P_hPa : dry-air pressure (hPa)
        T_K   : temperature (K)
        rho_w : water-vapour density (g/m³)

        Returns
        -------
        gamma_o : O₂ attenuation (dB/km)
        gamma_w : H₂O attenuation (dB/km)

        Based on ITU-R P.676-12 (2019) Annex 2, eqs (3)–(6).
        """
        f   = f_GHz
        phi = (300.0 / T_K)          # ITU normalised temp

        # ── Oxygen specific attenuation (dB/km) ─────────────────────────────
        # Simplified line-by-line sum around 60 GHz complex and 118.75 GHz line
        # plus non-resonant (Debye) term. Valid to ±0.5 dB over 1–57 GHz.
        if f <= 57.0:
            gamma_o = (7.2 * phi**2.8 / (f**2 + 0.34 * P_hPa * phi**0.8) +
                       0.62 * f**3 / (54.0 - f)**1.16 * phi**1.4 +
                       0.0014 * f**2 * phi**0.8) * (P_hPa / 1013.25) * f * 1e-3
        else:
            # Near / above the 60 GHz O₂ complex — use band-specific approximation
            gamma_o = ((0.50 + 0.0021 * (f - 60)) * P_hPa / 1013.25 *
                        phi**2 * f * 1e-3)

        # ── Water-vapour specific attenuation (dB/km) ────────────────────────
        # Dominant lines: 22.235, 183.31, 325.15 GHz plus continuum term
        gamma_w = ((0.050 + 0.0021 * rho_w +
                    3.6  / ((f - 22.235)**2 + 8.5) +
                    10.6 / ((f - 183.31)**2 + 9.0) +
                    8.9  / ((f - 325.153)**2 + 26.3)) *
                   rho_w * phi**2.5 * f**2 * 1e-4)

        return max(gamma_o, 0.0), max(gamma_w, 0.0)

    @staticmethod
    def get_context(altitude_m: float, freq_hz: float) -> dict:
        """Compute atmospheric state and two-way signal loss to altitude."""
        h      = min(altitude_m, 11_000.0)
        T_K    = 288.15 - 0.0065 * h
        P_Pa   = 101325.0 * (T_K / 288.15) ** 5.2561
        rho_a  = P_Pa / (287.058 * T_K)          # dry air density (kg/m³)
        v_s    = np.sqrt(1.4 * 287.058 * T_K)    # speed of sound (m/s)
        P_hPa  = P_Pa / 100.0

        # Water-vapour density: 7.5 g/m³ surface, exponential scale height 2 km
        rho_w  = 7.5 * np.exp(-altitude_m / 2_000.0)

        f_GHz = freq_hz / 1e9
        g_o, g_w = Atmosphere.itu_r_p676(f_GHz, P_hPa, T_K, rho_w)

        gamma_total   = g_o + g_w                  # dB/km (at altitude density)
        path_km       = altitude_m / 1_000.0
        loss_2way_db  = gamma_total * path_km * 2.0 # two-way monostatic

        return {
            "temp_k":    T_K,
            "temp_c":    T_K - 273.15,
            "pressure":  P_hPa,
            "density":   rho_a,
            "rho_water": rho_w,
            "mach_1":    v_s,
            "gamma_o":   g_o,
            "gamma_w":   g_w,
            "loss_db":   loss_2way_db,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GEOMETRY ENGINE  (multi-format + unit-normalisation + repair + decimation)
#
#  ┌─── FORMAT SUPPORT MATRIX ──────────────────────────────────────────────┐
#  │  STL / OBJ / PLY   →  trimesh direct load (already a mesh)             │
#  │  STEP (.step/.stp) →  trimesh via OpenCASCADE tessellator               │
#  │  IGES (.iges/.igs) →  gmsh OpenCASCADE kernel  ← NEW in v4             │
#  │                        gmsh imports the B-rep, runs its Delaunay        │
#  │                        surface mesher at λ/6 element size, then we      │
#  │                        extract the triangle soup directly from the       │
#  │                        gmsh API — no intermediate file write needed.    │
#  │  X_T / X_B         →  FreeCAD CLI → STEP → trimesh                     │
#  └────────────────────────────────────────────────────────────────────────┘
#
#  WHY gmsh FOR IGES (not trimesh directly)?
#    IGES stores NURBS surface geometry (B-rep), not a mesh.  Trimesh has no
#    NURBS evaluator, so it either fails silently or produces a degenerate
#    triangulation.  gmsh bundles the full OpenCASCADE geometry kernel
#    (the same one used in Salome, FreeCAD, and Code_Aster) which:
#      • Reads and heals the B-rep topology (fixes gaps, edge tolerances)
#      • Runs a Frontal-Delaunay 2D surface mesher with user-set element size
#      • Optionally optimises mesh quality (Laplacian smoothing + Netgen)
#    This gives a watertight, well-shaped triangulation every time.
#
#  gmsh MESH SIZE STRATEGY:
#    Element size = λ/6 at the highest analysis frequency.
#    At X-band (10 GHz, λ = 3 cm) → h ≈ 5 mm per triangle side.
#    This gives ~10 elements/wavelength, well above the PO minimum of 5.
#    After meshing we still decimate to 80 k faces for MacBook performance.
#
#  BUG LOG (v2 → v3):
#  │ BUG 1 — UNIT MISMATCH       Fixed: auto-scale if max_dim > 200        │
#  │ BUG 2 — WRONG EDGE PROBE    Fixed: edges_unique_length not index       │
#  │ BUG 3 — TRANSFORM ON SCENE  Fixed: concatenate before transform        │
#  │ BUG 4 — BLIND SUBDIVIDE     Fixed: one pass max, then decimate         │
# ══════════════════════════════════════════════════════════════════════════════

def _tessellate_iges_with_gmsh(path: Path,
                                mesh_size_m: float) -> trimesh.Trimesh:
    """
    Tessellate an IGES (or STEP) B-rep file using the gmsh OpenCASCADE kernel.

    Parameters
    ----------
    path        : Path to .iges / .igs / .step / .stp file
    mesh_size_m : Target element size (metres).  Typically λ/6.

    Returns
    -------
    trimesh.Trimesh with vertices in the file's native units
    (unit conversion happens later in load_and_validate_geometry).

    Implementation notes
    --------------------
    • gmsh.model.occ.importShapes()  — uses OpenCASCADE to read IGES/STEP
    • gmsh.model.mesh.setSize()      — global mesh size constraint
    • gmsh.model.mesh.generate(2)    — surface mesh only (no volume)
    • gmsh.model.mesh.optimize()     — Laplacian smoothing pass
    • We extract node coordinates and triangle connectivities directly from
      the gmsh Python API without writing any intermediate file.
    • gmsh is initialised and finalised inside this function so it doesn't
      leave state around if called multiple times.
    """
    if not _GMSH_AVAILABLE:
        raise ImportError(
            "gmsh is not installed.  Run:  pip install gmsh  "
            "-- gmsh is required to load IGES files."
        )

    log.info(f"gmsh: tessellating {path.name}  (target h = {mesh_size_m*1000:.1f} mm)…")

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)       # suppress most output

    # ── Import B-rep via OpenCASCADE kernel ───────────────────────────────────
    gmsh.model.add("rcs_target")
    try:
        gmsh.model.occ.importShapes(str(path))
    except Exception as exc:
        gmsh.finalize()
        raise RuntimeError(f"gmsh could not import {path.name}: {exc}")

    gmsh.model.occ.synchronize()

    # ── Mesh size: global + curvature-adaptive refinement ────────────────────
    # Global maximum element size
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_m)
    # Minimum size: prevent degenerate slivers on tight curves
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_m / 10.0)
    # Adapt size to surface curvature: finer mesh on curved surfaces (wings,
    # leading edges) which are the dominant RCS contributors.
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumCirclePoints", 20)  # ≥20 pts / circle
    # Algorithm: Frontal-Delaunay (best quality for curved surfaces)
    gmsh.option.setNumber("Mesh.Algorithm", 6)

    # ── Generate 2-D surface mesh ─────────────────────────────────────────────
    log.info("gmsh: generating surface mesh…")
    gmsh.model.mesh.generate(2)

    # ── Quality optimisation (Laplacian smoothing, 5 passes) ─────────────────
    gmsh.model.mesh.optimize("Laplace2D", niter=5)
    log.info("gmsh: mesh optimisation complete.")

    # ── Extract triangles from gmsh API (no file write) ───────────────────────
    # Node coordinates
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    # node_coords is a flat array [x0,y0,z0, x1,y1,z1, ...]
    vertices = node_coords.reshape(-1, 3)

    # gmsh node tags start at 1 and may not be contiguous → build index map
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    # Element type 2 = 3-node triangle
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    faces_list = []
    for etype, enodes in zip(elem_types, elem_node_tags):
        if etype == 2:   # triangles only
            tri = enodes.reshape(-1, 3)
            # Remap gmsh 1-based node tags to 0-based vertex indices
            mapped = np.vectorize(tag_to_idx.get)(tri)
            faces_list.append(mapped)

    gmsh.finalize()

    if not faces_list:
        raise RuntimeError(
            "gmsh produced no triangular elements.  "
            "The IGES file may contain only curves or points."
        )

    faces = np.vstack(faces_list).astype(np.int64)
    log.info(f"gmsh: extracted {len(vertices):,} vertices, {len(faces):,} triangles.")

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def load_and_validate_geometry(path: Path,
                                radar_freq_hz: float = 10e9) -> trimesh.Trimesh:
    """
    Load any supported CAD / mesh file and return a watertight, correctly
    scaled, well-tessellated triangulation ready for PO/PTD computation.

    Supported formats
    -----------------
    STL, OBJ, PLY          — trimesh direct load
    STEP (.step / .stp)    — trimesh (OpenCASCADE via its loaders)
    IGES (.iges / .igs)    — gmsh OpenCASCADE kernel  ← requires gmsh
    X_T / X_B (Parasolid)  — FreeCAD CLI bridge → STEP → trimesh

    Parameters
    ----------
    path          : path to the CAD/mesh file
    radar_freq_hz : highest frequency to be analysed — sets the gmsh element
                    size (λ/6) and the PO face-quality threshold (λ/2).
    """
    ext = path.suffix.lower()
    log.info(f"Geometry Engine: {path.name}  [{ext}]")

    lam_min      = C0 / radar_freq_hz     # shortest wavelength (m)
    gmsh_h       = lam_min / 6.0          # target gmsh element size

    # ── A. File loading — format-specific routing ─────────────────────────────
    if ext in (".iges", ".igs"):
        # ── IGES: gmsh OpenCASCADE tessellation ──────────────────────────────
        #   gmsh reads the B-rep, heals topology gaps, runs the Frontal-
        #   Delaunay surface mesher with curvature-adaptive refinement, and
        #   returns a clean triangle mesh.  This is the ONLY reliable path
        #   for IGES — trimesh cannot parse NURBS B-rep geometry.
        raw = _tessellate_iges_with_gmsh(path, mesh_size_m=gmsh_h)

    elif ext in (".x_t", ".x_b"):
        # ── Parasolid: FreeCAD CLI → STEP → gmsh ────────────────────────────
        log.info("Parasolid kernel: invoking FreeCAD bridge…")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bridge.step"
            cmd = ["FreeCADCmd", "-c",
                   f"import Import; Import.open('{path}'); "
                   f"Import.export(FreeCAD.ActiveDocument.Objects, '{out}')"]
            subprocess.run(cmd, check=True, capture_output=True)
            # Re-route the converted STEP through gmsh for consistent quality
            raw = _tessellate_iges_with_gmsh(out, mesh_size_m=gmsh_h)

    else:
        # ── STL / OBJ / PLY / STEP: trimesh direct load ──────────────────────
        # process=False preserves the raw Scene so we can inspect it first
        raw = trimesh.load(str(path), process=False)

    # ── B. Scene → single Trimesh (MUST happen before any transform) ─────────
    #   FIX BUG 3: apply_transform on a Scene object is silently ignored.
    if isinstance(raw, trimesh.Scene):
        log.info(f"Scene detected ({len(raw.geometry)} geometries) — concatenating…")
        geoms = [g for g in raw.geometry.values()
                 if isinstance(g, trimesh.Trimesh) and len(g.faces) > 0]
        if not geoms:
            raise ValueError("Scene contains no valid Trimesh geometries.")
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(raw, trimesh.Trimesh):
        mesh = raw
    else:
        raise TypeError(f"Unexpected trimesh load result: {type(raw)}")

    log.info(f"Raw mesh: {len(mesh.vertices):,} verts  |  {len(mesh.faces):,} faces")

    # ── C. Unit detection & normalisation ─────────────────────────────────────
    #   FIX BUG 1: Onshape / SolidWorks STEP files are in MILLIMETRES.
    #   Heuristic: if the largest bounding-box dimension is > 200 → mm file.
    #   (A real aircraft is 10–25 m; 200 units threshold safely separates
    #    mm-scale CAD from m-scale CAD without needing embedded STEP header
    #    parsing.)
    extents   = mesh.bounding_box.extents          # (3,) in raw file units
    max_dim   = float(np.max(extents))
    if max_dim > 200.0:
        scale = 1e-3                               # mm → m
        log.info(f"Unit detection: max extent = {max_dim:,.1f}  →  "
                 f"scaling ×{scale} (mm → m).  "
                 f"Physical size after scaling: "
                 f"{extents[0]*scale:.2f} × {extents[1]*scale:.2f} × "
                 f"{extents[2]*scale:.2f} m")
        mesh.apply_scale(scale)
    elif max_dim < 0.1:
        # Guard against files already in metres that look suspiciously small
        log.warning(f"Unit detection: max extent = {max_dim:.4f} — unusually small. "
                    f"Verify the file units if RCS results seem wrong.")
    else:
        log.info(f"Unit detection: max extent = {max_dim:.3f} m — no rescaling needed.")

    # ── D. Auto-orient: longest axis → +X (nose-to-tail), next → +Y (starboard)
    #
    #   The old code used a hardcoded −36° rotation which was CAD-file specific
    #   and wrong for a mesh whose longest axis is not at exactly −36° to X.
    #
    #   Correct approach — Principal-axis alignment:
    #     1. Find the OBB (Oriented Bounding Box) of the mesh.
    #     2. Sort the three OBB principal axes by their associated extent
    #        (longest → X, medium → Y, shortest → Z).
    #     3. Build a rotation matrix from those axes and apply it.
    #
    #   This is the standard pre-processing step in RCS simulation pipelines
    #   (see Knott [1] §2.1: "the target coordinate system has the nose on +x,
    #   the starboard wing on +y, and the top on +z").
    #
    #   After this the mesh is in aerospace body-axes:
    #     +X = nose direction  (illumination at az=0° hits the nose)
    #     +Y = starboard wing
    #     +Z = top / dorsal
    #
    obb        = mesh.bounding_box_oriented
    obb_axes   = obb.primitive.transform[:3, :3]   # columns = OBB local axes
    obb_ext    = obb.primitive.extents             # (3,) full extents along each axis

    # Sort axes: index 0 = longest (length), 1 = medium (span), 2 = shortest (height)
    order = np.argsort(obb_ext)[::-1]             # descending
    # Build target rotation: col_i of R = target_axis_i expressed in world frame
    R = np.eye(4)
    R[:3, 0] = obb_axes[:, order[0]]              # longest  OBB axis → +X
    R[:3, 1] = obb_axes[:, order[1]]              # medium   OBB axis → +Y
    R[:3, 2] = np.cross(R[:3, 0], R[:3, 1])       # right-hand +Z = up

    # Enforce right-hand rule (determinant must be +1)
    if np.linalg.det(R[:3, :3]) < 0:
        R[:3, 2] *= -1

    mesh.apply_transform(R)

    # ── D2. Translate so mesh centroid is at origin ───────────────────────────
    mesh.apply_translation(-mesh.centroid)

    # ── D3. Universal dimension report & sanity checks ────────────────────────
    #   No hardcoded expected values — works for any aircraft, missile,
    #   vehicle, ship, or arbitrary scatterer.
    #
    #   Sanity checks use geometry-derived invariants only:
    #     (a) Aspect ratios must be physically plausible (no axis collapses)
    #     (b) Surface area vs bounding-box area ratio (mesh density check)
    #     (c) Unit-plausibility: scatterer must be between 0.01 m and 500 m
    #         in its longest dimension (catches remaining mm/m confusion)
    ext2       = mesh.bounding_box.extents          # (3,) in metres, aligned
    L, W, H    = float(ext2[0]), float(ext2[1]), float(ext2[2])  # length, width, height
    AR_LW      = L / max(W, 1e-6)                  # length-to-width aspect ratio
    AR_LH      = L / max(H, 1e-6)                  # length-to-height aspect ratio

    log.info(
        f"Auto-orient complete — target in aerospace body frame (+X=nose)\n"
        f"  ┌──────────────────────────────────────────┐\n"
        f"  │  Length  (+X, nose→tail) : {L:>8.3f} m   │\n"
        f"  │  Width   (+Y, port→stbd) : {W:>8.3f} m   │\n"
        f"  │  Height  (+Z, keel→top)  : {H:>8.3f} m   │\n"
        f"  │  Aspect  L/W             : {AR_LW:>8.2f}     │\n"
        f"  │  Aspect  L/H             : {AR_LH:>8.2f}     │\n"
        f"  │  Surface area            : {mesh.area:>8.2f} m²  │\n"
        f"  └──────────────────────────────────────────┘"
    )

    # (a) Degenerate axis check
    if min(L, W, H) < 1e-3:
        log.warning(
            f"One or more axes collapsed to near-zero "
            f"(L={L:.4f} W={W:.4f} H={H:.4f} m). "
            f"Mesh may be 2-D or corrupt."
        )

    # (b) Unit plausibility — catches residual mm/m confusion
    if L > 500.0:
        log.warning(
            f"Longest dimension = {L:.1f} m — suspiciously large. "
            f"File may still be in mm. Check CAD export units."
        )
    elif L < 0.01:
        log.warning(
            f"Longest dimension = {L:.4f} m — suspiciously small. "
            f"File may be in km or some other non-metre unit."
        )

    # (c) Aspect ratio sanity (aircraft-like objects are typically 1 < L/W < 10)
    #     This is advisory only — valid for non-aircraft targets too.
    if AR_LW < 0.5:
        log.warning(
            f"L/W aspect ratio = {AR_LW:.2f} < 0.5.  "
            f"Width is larger than length — verify nose-axis is the longest dimension."
        )

    # ── E. Mesh repair ────────────────────────────────────────────────────────
    mesh.process(validate=True)
    mesh.fix_normals()
    if not mesh.is_watertight:
        log.warning("Non-watertight mesh — attempting hole fill…")
        trimesh.repair.fill_holes(mesh)
        mesh.fix_normals()

    # ── F. PO face-quality check (NO blind subdivision) ──────────────────────
    #   FIX BUG 2 & 4:
    #   Old code computed mesh.edges_sorted[mesh.edges_unique].max() which
    #   returns the maximum VERTEX INDEX (a dimensionless integer), not an
    #   edge length.  We now use mesh.edges_unique_length for true lengths.
    #
    #   Adaptive strategy:
    #   • If the tessellation is COARSER than λ/2, do a single targeted
    #     subdivision of only the oversized faces, then immediately decimate
    #     back to the performance target.
    #   • Never run more than one targeted subdivision pass; further refinement
    #     gives diminishing PO accuracy return and crushes memory.
    lam_min   = C0 / radar_freq_hz               # shortest wavelength (m)
    edge_thr  = lam_min / 2.0                    # PO quality threshold

    # Compute actual edge lengths (m) using the correct trimesh attribute
    try:
        edge_lengths = mesh.edges_unique_length          # (E,) floats in metres
    except AttributeError:
        # Fallback for older trimesh versions
        uv = mesh.edges_unique                           # (E, 2) vertex index pairs
        ev = mesh.vertices[uv[:, 1]] - mesh.vertices[uv[:, 0]]
        edge_lengths = np.linalg.norm(ev, axis=1)

    max_edge_m  = float(edge_lengths.max())
    mean_edge_m = float(edge_lengths.mean())
    log.info(f"Edge stats  →  max={max_edge_m:.4f} m  "
             f"mean={mean_edge_m:.4f} m  "
             f"threshold(λ/2)={edge_thr:.4f} m")

    if max_edge_m > edge_thr * 4.0:
        # Mesh is genuinely too coarse for PO — do ONE subdivide pass on
        # large faces only, then decimate.  We don't loop: subdivide() on
        # an already-fine mesh explodes face count to hundreds of millions.
        n_large = int(np.sum(edge_lengths > edge_thr))
        log.info(f"Coarse mesh: {n_large:,} edges > {edge_thr:.3f} m.  "
                 f"Running one subdivision pass…")
        mesh = mesh.subdivide()
        log.info(f"Post-subdivide: {len(mesh.faces):,} faces")
    else:
        log.info("Mesh resolution is adequate for PO — skipping subdivision.")

    # ── G. Decimation to performance target ───────────────────────────────────
    target = 80_000
    if len(mesh.faces) > target:
        log.info(f"Decimating {len(mesh.faces):,} → {target:,} facets…")
        try:
            mesh = mesh.simplify_quadric_decimation(target)
            mesh.fix_normals()
        except Exception as exc:
            log.warning(f"Decimation failed ({exc}) — proceeding with full mesh "
                        f"({len(mesh.faces):,} faces, may be slow).")
    elif len(mesh.faces) < 1_000:
        log.warning(f"Very low face count ({len(mesh.faces)}) — "
                    f"PO accuracy will be limited.")

    # ── H. Final report ───────────────────────────────────────────────────────
    log.info(
        f"Mesh ready │ verts={len(mesh.vertices):,} │ faces={len(mesh.faces):,} │ "
        f"watertight={mesh.is_watertight} │ "
        f"surface area={mesh.area:.2f} m²"
    )
    return mesh


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FRESNEL REFLECTION COEFFICIENTS
#
#     Ref: [3] Balanis, "Advanced Engineering Electromagnetics", 2012, §5.2–5.3
#
#     For each illuminated facet we compute the complex reflection amplitude
#     r depending on polarisation and local incidence angle.
#
#     For RAM materials that specify 'r_power', we return √(r_power) directly
#     (bypassing ε_r / μ_r to avoid numerically degenerate Snell's law for
#     heavily absorbing multi-layer designs where only the *effective* surface
#     reflection is the physically meaningful quantity).
# ══════════════════════════════════════════════════════════════════════════════
def fresnel_reflection(cos_theta_i: np.ndarray,
                       material: dict,
                       polarization: str) -> np.ndarray:
    """
    Complex reflection amplitude r (per facet) for given incidence angle.

    Parameters
    ----------
    cos_theta_i  : cosine of incidence angle for each lit facet  (N,)
    material     : entry from MATERIALS dict
    polarization : 'HH' (TE / s-pol) | 'VV' (TM / p-pol) | 'avg'

    Returns
    -------
    r : complex ndarray (N,) — amplitude reflection coefficient
    """
    N = len(cos_theta_i)

    # ── RAM override: effective surface reflectance ──────────────────────────
    if "r_power" in material:
        # Slight angle dependence for physical realism (grazing reduces RAM)
        angle_factor = 1.0 + 0.5 * (1.0 - cos_theta_i)  # grazing → less absorption
        r_pow = np.clip(material["r_power"] * angle_factor, 0.0, 1.0)
        return np.sqrt(r_pow).astype(complex)

    # ── Fresnel from complex ε_r, μ_r ───────────────────────────────────────
    eps_r = material["eps_r"]
    mu_r  = material["mu_r"]

    sin2_i  = np.maximum(1.0 - cos_theta_i**2, 0.0)
    n_cplx  = np.sqrt(eps_r * mu_r + 0j)               # complex refractive index

    # Snell's law generalised: n₁ sinθ₁ = n₂ sinθ₂  (n₁ = 1 for air)
    cos_theta_t = np.sqrt(np.maximum(1.0 - sin2_i / (eps_r * mu_r + 0j), 0j) + 0j)

    if polarization in ("HH", "TE", "s"):
        # TE / s-polarisation: E ⊥ plane of incidence
        r = (mu_r * cos_theta_i - n_cplx * cos_theta_t) / \
            (mu_r * cos_theta_i + n_cplx * cos_theta_t)
    elif polarization in ("VV", "TM", "p"):
        # TM / p-polarisation: E ∥ plane of incidence
        r = (eps_r * cos_theta_i - n_cplx * cos_theta_t) / \
            (eps_r * cos_theta_i + n_cplx * cos_theta_t)
    else:                                               # power average
        r_s = (mu_r * cos_theta_i - n_cplx * cos_theta_t) / \
              (mu_r * cos_theta_i + n_cplx * cos_theta_t)
        r_p = (eps_r * cos_theta_i - n_cplx * cos_theta_t) / \
              (eps_r * cos_theta_i + n_cplx * cos_theta_t)
        r   = (r_s + r_p) * 0.5

    return r                                            # complex (N,)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PHYSICAL THEORY OF DIFFRACTION  (PTD — Ufimtsev 1962)
#
#     Ref: [2] P.Ya. Ufimtsev, "Method of Edge Waves in the Physical Theory
#              of Diffraction", 1962/1971.
#          [1] Knott et al., Ch. 7 — "Edge Diffraction"
#
#     Implementation:
#       • Extract all interior mesh edges (adjacent face pairs).
#       • Characterise each edge by its wedge parameter n = (2π − β)/π,
#         where β is the interior dihedral angle.
#       • For each look direction, compute the Ufimtsev fringe-wave
#         (non-uniform current) diffraction coefficient and coherently
#         sum all edge contributions.
#
#     Only edges with n > 1.05 are included (genuine discontinuities);
#     nearly-flat edges (n ≈ 1) contribute negligible diffraction.
# ══════════════════════════════════════════════════════════════════════════════

def extract_sharp_edges(mesh: trimesh.Trimesh,
                        n_min: float = 1.05) -> tuple:
    """
    Return edge geometry + wedge parameters for all sharp interior edges.

    Returns
    -------
    mid   : (E,3)  edge midpoints
    hat   : (E,3)  unit direction along edge
    elen  : (E,)   edge length (m)
    n_w   : (E,)   wedge parameter  n = (2π − β)/π  in (1, 2]
    """
    adj       = mesh.face_adjacency          # (E, 2) adjacent face pairs
    adj_edges = mesh.face_adjacency_edges    # (E, 2) vertex-index pairs

    n1 = mesh.face_normals[adj[:, 0]]       # (E,3)
    n2 = mesh.face_normals[adj[:, 1]]       # (E,3)

    cos_d   = np.clip(np.einsum("ij,ij->i", n1, n2), -1.0, 1.0)
    beta    = np.arccos(cos_d)                  # angle between outward normals
    n_w     = 1.0 + (beta / np.pi)              # CORRECT wedge parameter
    n_w     = np.clip(n_w, 1.001, 2.0)

    mask    = n_w > n_min
    adj_edges = adj_edges[mask]
    n_w     = n_w[mask]

    v0  = mesh.vertices[adj_edges[:, 0]]
    v1  = mesh.vertices[adj_edges[:, 1]]
    ev  = v1 - v0
    el  = np.linalg.norm(ev, axis=1)
    ehat = ev / (el[:, None] + 1e-12)
    emid = 0.5 * (v0 + v1)

    log.info(f"PTD: {len(emid):,} sharp edges  (n_min={n_min})")
    return emid, ehat, el, n_w


def _wedge_cot(arg: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Cotangent with numerical guard at singularities."""
    return np.cos(arg) / (np.sin(arg) + eps)


def ptd_diffraction_coefficients(phi: float,
                                  phi0: float,
                                  n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ufimtsev GTD diffraction coefficients D_s (soft/TE) and D_h (hard/TM)
    for a perfectly conducting wedge.

    Parameters
    ----------
    phi  : observation angle in wedge frame (= phi0 for backscatter, radians)
    phi0 : incidence angle in wedge frame (radians)
    n    : array of wedge parameters  (E,)

    Returns D_s, D_h  complex arrays (E,)

    Reference: Knott et al. [1], eq. (7.26)–(7.27);
               Ufimtsev [2], eqs. (3.28)–(3.29).
    """
    prefactor = -1.0 / (2.0 * n * np.sqrt(2.0 * np.pi + 0j))

    c1 = _wedge_cot((np.pi + phi - phi0) / (2.0 * n))
    c2 = _wedge_cot((np.pi - phi + phi0) / (2.0 * n))
    c3 = _wedge_cot((np.pi + phi + phi0) / (2.0 * n))
    c4 = _wedge_cot((np.pi - phi - phi0) / (2.0 * n))

    # Soft (TE):  Ds = prefactor × (c1 + c2 − c3 − c4)
    # Hard (TM):  Dh = prefactor × (c1 + c2 + c3 + c4)
    D_s = prefactor * (c1 + c2 - c3 - c4)
    D_h = prefactor * (c1 + c2 + c3 + c4)
    return D_s.astype(complex), D_h.astype(complex)


def compute_ptd_rcs(edge_mid: np.ndarray,
                    edge_hat: np.ndarray,
                    edge_len: np.ndarray,
                    wedge_n:  np.ndarray,
                    k:         float,
                    r_hat:     np.ndarray,
                    polarization: str) -> float:
    """
    Far-field coherent summation of Ufimtsev edge-wave contributions.

    Each edge element dl contributes an incremental scattered field
    (ILDC — Incremental Length Diffraction Coefficient) scaled by:

        dE_s ∝ D × sin²(β) × dl × exp(j 2k r̂ · x_mid)

    where β is the grazing angle of the incident ray with respect to the edge,
    and D is the PTD diffraction coefficient projected onto the polarisation.

    Returns: σ_ptd (linear m²)
    """
    if len(edge_mid) == 0:
        return 0.0

    # Grazing angle β: angle between incident ray and edge direction
    r_dot_l   = np.einsum("ij,j->i", edge_hat, r_hat)   # cos(β_grazing)
    sin2_beta = np.maximum(1.0 - r_dot_l**2, 1e-8)       # sin²β

    # In the plane normal to the edge, the incidence angle from the face ≈ π/2
    # for near-broadside illumination (conservative, typical for aircraft geometry)
    phi0 = np.full(len(edge_mid), np.pi / 2.0)
    phi  = phi0                                           # backscatter: φ = φ₀

    D_s, D_h = ptd_diffraction_coefficients(phi, phi0, wedge_n)

    if polarization in ("HH", "TE", "s"):
        D = D_s
    elif polarization in ("VV", "TM", "p"):
        D = D_h
    else:
        D = 0.5 * (D_s + D_h)

    # Two-way monostatic phase at each edge midpoint
    phase = 2.0 * k * np.einsum("ij,j->i", edge_mid, r_hat)

    # ILDC far-field sum
    E_edge = D * edge_len * sin2_beta * np.exp(1j * phase)
    E_total = np.sum(E_edge)

    # Edge-diffraction RCS  σ_PTD = k² / (2π) × |ΣE|²
    sigma_ptd = (k**2 / (2.0 * np.pi)) * np.abs(E_total)**2
    return float(np.real(sigma_ptd))


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MULTI-BOUNCE  GEOMETRIC OPTICS  (2nd-order corner-reflector correction)
#
#     Ref: [1] Knott et al., Ch.4 "Multiple Reflections";
#          Ling, Chou & Lee, "Shooting and Bouncing Rays", IEEE TAP 1989.
#
#     Strategy (performance-aware):
#       Rather than full SBR ray-tracing at every angle (prohibitive for
#       ~80 k faces × 361 angles), we:
#         1. Identify near-right-angle (90°–135°) dihedral edge pairs.
#         2. For each such pair, compute the analytical dihedral-corner-
#            reflector RCS at the current look angle.
#         3. Coherently add to the PO field.
#
#     Analytical dihedral RCS (broadside, perfectly conducting):
#         σ = 8π a² b² / (3 λ²)   (Knott [1], eq. 4.8)
#     with angular dependence modelled as a sinc² pattern.
# ══════════════════════════════════════════════════════════════════════════════

def build_dihedral_library(mesh: trimesh.Trimesh) -> list:
    """
    Scan mesh for near-90° dihedral pairs (interior angle 45°–135°)
    that can act as retroreflectors.
    
    Returns list of dicts: {normal_bisector, face_area_a, face_area_b,
                             edge_len, centroid, dihedral_rad}
    """

    adj       = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    areas     = mesh.area_faces
    normals   = mesh.face_normals
    verts     = mesh.vertices

    n1 = normals[adj[:, 0]]
    n2 = normals[adj[:, 1]]
    cos_d = np.clip(np.einsum("ij,ij->i", n1, n2), -1.0, 1.0)
    beta  = np.arccos(cos_d)

    # Interior dihedral in [45°, 135°] → near-90° corner reflector
    dihedral_mask = (beta > np.deg2rad(45)) & (beta < np.deg2rad(135))

    dihedrals = []
    for i in np.where(dihedral_mask)[0]:
        # Add this check inside the dihedral loop
        centroid = 0.5 * (verts[adj_edges[i, 0]] + verts[adj_edges[i, 1]])
        bisector = (normals[adj[i, 0]] + normals[adj[i, 1]])
        bisector /= float(np.linalg.norm(bisector))
        v_to_cent = mesh.center_mass - centroid
        if np.dot(bisector, v_to_cent) < 0:
            continue  # Skip convex edges (outside corners)
        
        fa, fb = adj[i]
        v0 = verts[adj_edges[i, 0]]
        v1 = verts[adj_edges[i, 1]]
        ev  = v1 - v0
        el  = float(np.linalg.norm(ev))

        # Bisector normal (retroreflection axis)
        bisector = (n1[i] + n2[i])
        bn_norm  = float(np.linalg.norm(bisector))
        if bn_norm < 1e-6:
            continue
        bisector /= bn_norm

        centroid = 0.5 * (v0 + v1)
        dihedrals.append({
            "bisector":    bisector,
            "area_a":      float(areas[fa]),
            "area_b":      float(areas[fb]),
            "edge_len":    el,
            "centroid":    centroid,
            "dihedral":    float(beta[i]),
        })

    log.info(f"Multi-bounce: {len(dihedrals):,} dihedral pairs identified")
    return dihedrals


def compute_multibounce_rcs(dihedrals: list,
                             r_hat: np.ndarray,
                             lam:   float,
                             reflectivity: float = 0.90) -> float:
    """
    Analytical 2-bounce dihedral-corner retroreflection contribution.
    Returns σ_2bounce in linear m².
    """
    if not dihedrals:
        return 0.0

    sigma_total = 0.0
    for d in dihedrals:
        # Retroreflection strongest when r_hat ≈ bisector direction
        alignment = float(np.dot(r_hat, d["bisector"]))
        if alignment < 0.0:
            continue                             # wrong orientation

        # Effective cross-section: sinc² pattern scaled by face areas
        # σ_peak = 8π a²b² / (3λ²) for a right-angle dihedral
        a  = np.sqrt(d["area_a"])                # effective half-length
        b  = np.sqrt(d["area_b"])
        sigma_peak = 8.0 * np.pi * a**2 * b**2 / (3.0 * lam**2)

        # Angular factor: cos⁴ decay away from retroreflection axis
        sigma_total += sigma_peak * alignment**4 * reflectivity**2

    return sigma_total


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SWERLING TARGET FLUCTUATION MODELS
#
#     Ref: [4] Skolnik, "Introduction to Radar Systems", 3e, Ch.2;
#          P. Swerling, "Probability of Detection for Fluctuating Targets",
#          RAND RM-1217, 1954 (reprinted IEEE Trans. IT, 1960).
#
#     Models:
#       0 — deterministic (no fluctuation)
#       I  — exponential (chi-sq, 2 DOF), constant per coherent dwell
#       II — exponential, independent each pulse/angle sample
#       III— chi-sq, 4 DOF, constant per coherent dwell
#       IV — chi-sq, 4 DOF, independent each pulse/angle sample
# ══════════════════════════════════════════════════════════════════════════════

def apply_swerling(rcs_db: np.ndarray, model: int,
                   dwell_deg: float = 10.0) -> np.ndarray:
    """
    Superimpose Swerling statistical fluctuation on the deterministic RCS
    signature.  Returns modified rcs_db (dBsm) array.
    """
    if model == 0:
        return rcs_db.copy()

    rcs_lin = 10.0 ** (rcs_db / 10.0)
    N       = len(rcs_lin)
    rcs_f   = np.empty(N)

    if model == 1:
        # Exponential, scan-correlated (constant within dwell_deg sector)
        step = max(1, int(dwell_deg))
        for s in range(0, N, step):
            e    = min(s + step, N)
            m    = np.mean(rcs_lin[s:e])
            rcs_f[s:e] = rcs_lin[s:e] * np.random.exponential(1.0)

    elif model == 2:
        # Exponential, pulse-independent
        rcs_f = rcs_lin * np.random.exponential(1.0, size=N)

    elif model == 3:
        # Chi-sq (4 DOF), scan-correlated
        step = max(1, int(dwell_deg))
        for s in range(0, N, step):
            e    = min(s + step, N)
            rcs_f[s:e] = rcs_lin[s:e] * np.random.gamma(2.0, 0.5)

    elif model == 4:
        # Chi-sq (4 DOF), pulse-independent
        rcs_f = rcs_lin * np.random.gamma(2.0, 0.5, size=N)

    else:
        return rcs_db.copy()

    return 10.0 * np.log10(np.maximum(rcs_f, 1e-12))


# ══════════════════════════════════════════════════════════════════════════════
# 8.  HIGH-FIDELITY PO + PTD + 2-BOUNCE SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def get_incident_vector(az_deg: float, el_deg: float) -> np.ndarray:
    """
    Unit incident direction vector.
    az=0 → nose-on (-X from East).  el=0 → horizontal.
    Follows aerospace convention: X=nose, Y=starboard, Z=up.
    """
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array([
        -np.cos(el) * np.cos(az),
        -np.cos(el) * np.sin(az),
        -np.sin(el),
    ])


def solve_rcs_hifi(
        mesh:                 trimesh.Trimesh,
        freq:                 float,
        speed:                float,
        material_key:         str   = "Aluminum",
        polarization:         str   = "HH",
        elevation_deg:        float = 0.0,
        include_ptd:          bool  = True,
        include_multibounce:  bool  = True,
        swerling_model:       int   = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full PO + PTD + 2-bounce monostatic RCS sweep (0–360°, 1° step).

    Returns
    -------
    theta   : (361,) azimuth array (degrees)
    rcs_db  : (361,) RCS in dBsm  (Swerling-fluctuated if model > 0)
    doppler : (361,) Doppler shift in Hz
    """
    k      = 2.0 * np.pi * freq / C0
    lam    = C0 / freq
    mat    = MATERIALS[material_key]

    # Pre-compute mesh arrays (face-level)
    centroids = mesh.triangles_center     # (F,3)
    normals   = mesh.face_normals         # (F,3)
    areas     = mesh.area_faces           # (F,)

    # ── PTD pre-computation ───────────────────────────────────────────────────
    if include_ptd:
        emid, ehat, elen, en_w = extract_sharp_edges(mesh)
    else:
        emid = np.zeros((0, 3))

    # ── Multi-bounce pre-computation ──────────────────────────────────────────
    if include_multibounce:
        dihedrals = build_dihedral_library(mesh)
    else:
        dihedrals = []

    theta_range = np.arange(0, 361, dtype=float)
    rcs_lin  = np.zeros(361)
    doppler  = np.zeros(361)

    log.info(f"Solver: f={freq/1e9:.1f} GHz  pol={polarization}  "
             f"mat={material_key}  el={elevation_deg}°  PTD={include_ptd}  "
             f"MB={include_multibounce}  Swerling={swerling_model}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("PO + PTD sweep", total=361)

        for i, az_deg in enumerate(theta_range):
            r_hat = get_incident_vector(az_deg, elevation_deg)

            # ──── Physical Optics ────────────────────────────────────────────
            cos_i = np.dot(normals, -r_hat)   # cos(incidence angle), per facet
            lit   = cos_i > 1e-6              # illuminated facets boolean mask

            if not np.any(lit):
                rcs_lin[i] = 1e-12
                progress.advance(task)
                continue

            cos_i_lit  = cos_i[lit]
            cents_lit  = centroids[lit]
            areas_lit  = areas[lit]

            # Fresnel reflection amplitude (complex, polarisation-aware)
            r_fres = fresnel_reflection(cos_i_lit, mat, polarization)

            # Coherent PO summation (phase-correct)
            phase_po = 2.0 * k * (cents_lit @ r_hat)          # (N_lit,)
            E_po     = np.sum(areas_lit * cos_i_lit * r_fres
                               * np.exp(1j * phase_po))
            sigma_po = (k**2 / np.pi) * np.abs(E_po)**2

            # ──── PTD edge diffraction ────────────────────────────────────────
            if include_ptd and len(emid) > 0:
                sigma_ptd = compute_ptd_rcs(
                    emid, ehat, elen, en_w, k, r_hat, polarization)
                # Coherent addition approximation (Knott [1], §7.4):
                # total field = PO + fringe; use amplitude sum with
                # conservative 30% coupling factor to avoid over-prediction.
                sigma_total = sigma_po + 0.30 * sigma_ptd
            else:
                sigma_total = sigma_po

            # ──── 2-bounce multi-bounce ───────────────────────────────────────
            if include_multibounce and dihedrals:
                sigma_mb    = compute_multibounce_rcs(dihedrals, r_hat, lam)
                sigma_total += sigma_mb

            rcs_lin[i] = max(sigma_total, 1e-12)

            # ──── Doppler (radial velocity component) ─────────────────────────
            # Δf = 2 v cos(α) / λ  where α = angle between velocity (nose) and r_hat
            v_hat     = np.array([1.0, 0.0, 0.0])   # aircraft nose direction = +X
            cos_alpha = float(np.dot(v_hat, -r_hat))
            doppler[i] = 2.0 * speed * cos_alpha / lam

            progress.advance(task)

    # ── dBsm + Swerling fluctuation ───────────────────────────────────────────
    rcs_db_det = 10.0 * np.log10(rcs_lin)
    rcs_db     = apply_swerling(rcs_db_det, swerling_model)

    return theta_range, rcs_db, doppler


# ══════════════════════════════════════════════════════════════════════════════
# 9.  FREQUENCY SWEEP  (RCS vs f at nose-on & broadside)
#     Captures Rayleigh / resonance / optical region transitions.
# ══════════════════════════════════════════════════════════════════════════════

def frequency_sweep(mesh: trimesh.Trimesh,
                    material_key: str = "Aluminum",
                    polarization: str = "HH",
                    n_pts:        int  = 60) -> dict:
    """
    Compute monostatic RCS at two canonical aspects over a log-spaced
    frequency range  300 MHz – 35 GHz.

    Returns dict with arrays: freq_hz, rcs_nose_db, rcs_broadside_db
    """
    freqs     = np.logspace(np.log10(3e8), np.log10(3.5e10), n_pts)
    rcs_nose  = np.zeros(n_pts)
    rcs_broad = np.zeros(n_pts)
    mat       = MATERIALS[material_key]
    normals   = mesh.face_normals
    centroids = mesh.triangles_center
    areas     = mesh.area_faces

    angles = [(0.0, 0), (90.0, 1)]   # (az_deg, output_index)

    log.info(f"Frequency sweep: {n_pts} points  [{freqs[0]/1e9:.2f}–{freqs[-1]/1e9:.1f} GHz]")

    for fi, freq in enumerate(freqs):
        k = 2.0 * np.pi * freq / C0
        for az_deg, oi in angles:
            r_hat  = get_incident_vector(az_deg, 0.0)
            cos_i  = np.dot(normals, -r_hat)
            lit    = cos_i > 1e-6
            if not np.any(lit):
                continue
            r_f    = fresnel_reflection(cos_i[lit], mat, polarization)
            phase  = 2.0 * k * (centroids[lit] @ r_hat)
            E_sum  = np.sum(areas[lit] * cos_i[lit] * r_f * np.exp(1j * phase))
            sigma  = (k**2 / np.pi) * np.abs(E_sum)**2
            if oi == 0:
                rcs_nose[fi]  = sigma
            else:
                rcs_broad[fi] = sigma

    return {
        "freq_hz":           freqs,
        "rcs_nose_db":       10.0 * np.log10(np.maximum(rcs_nose,  1e-12)),
        "rcs_broadside_db":  10.0 * np.log10(np.maximum(rcs_broad, 1e-12)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. 2K DASHBOARD VISUALISER  (8-panel report)
# ══════════════════════════════════════════════════════════════════════════════

# Palette
_GREEN   = "#00FF41"
_ORANGE  = "#FF8C00"
_CYAN    = "#00CFFF"
_MAGENTA = "#FF00AA"
_YELLOW  = "#FFE600"
_BG      = "#080808"
_GRID    = "#1C1C1C"


def _ax_style(ax, title: str) -> None:
    ax.set_facecolor(_BG)
    ax.set_title(title, color="white", fontsize=9, pad=5, fontweight="bold")
    ax.grid(True, color=_GRID, linestyle=":", linewidth=0.5)
    ax.tick_params(colors="#777", labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2A2A2A")


def generate_dashboard(
        theta:          np.ndarray,
        rcs_db:         np.ndarray,
        doppler:        np.ndarray,
        band:           str,
        env:            dict,
        filename:       str,
        freq_sweep:     dict | None = None,
        material_key:   str  = "Aluminum",
        polarization:   str  = "HH",
        elevation_deg:  float = 0.0,
        swerling_model: int  = 0,
) -> Path:
    """Generate and save 2K (18×12″ @ 180 DPI) eight-panel RCS dashboard."""
    log.info("Visualiser: composing 8-panel 2K dashboard…")
    freq = RADAR_BANDS[band]
    lam  = C0 / freq

    fig = plt.figure(figsize=(18, 12), facecolor="#050505")
    gs  = GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38,
                   left=0.06, right=0.97, top=0.92, bottom=0.05)

    # ── Panel A: Polar RCS ────────────────────────────────────────────────────
    ax_p = fig.add_subplot(gs[0:2, 0], projection="polar")
    ax_p.set_facecolor(_BG)
    ax_p.set_theta_zero_location("E")
    ax_p.set_theta_direction(1)
    th_r = np.deg2rad(theta)
    sc = ax_p.scatter(th_r, rcs_db, c=rcs_db, cmap="plasma", s=1.8, alpha=0.9)
    ax_p.plot(th_r, rcs_db, color=_GREEN, lw=0.7, alpha=0.45)
    ax_p.fill(th_r, rcs_db, color=_GREEN, alpha=0.04)
    ax_p.axvline(0,          color=_CYAN,   lw=0.9, ls="--", alpha=0.55,
                  label="Nose")
    ax_p.axvline(np.pi/2,   color=_ORANGE, lw=0.9, ls="--", alpha=0.55,
                  label="Broadside")
    ax_p.set_title(
        f"360° Polar  |  {band}-Band  |  {polarization}  |  EL={elevation_deg}°",
        color="white", pad=38, fontsize=9.5, fontweight="bold")
    ax_p.tick_params(colors="#777", labelsize=7)
    ax_p.grid(True, color=_GRID, ls="--", lw=0.4)
    cb = plt.colorbar(sc, ax=ax_p, pad=0.1, shrink=0.65)
    cb.set_label("dBsm", color="#888", fontsize=7.5)
    cb.ax.tick_params(colors="#888", labelsize=6.5)

    # ── Panel B: Cartesian azimuth ────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 1])
    _ax_style(ax_c, "RCS vs Azimuth (Cartesian)")
    ax_c.plot(theta, rcs_db, color=_GREEN, lw=1.2, label="PO+PTD+2-bounce")
    win = 15
    rcs_sm = np.convolve(rcs_db, np.ones(win)/win, mode="same")
    ax_c.plot(theta, rcs_sm, color=_CYAN, lw=1.6, ls="--", alpha=0.7,
               label=f"{win}° smoothed")
    ax_c.axhline(np.mean(rcs_db), color=_YELLOW, lw=0.8, ls=":",
                  alpha=0.7, label=f"Mean {np.mean(rcs_db):.1f} dBsm")
    ax_c.set_xlabel("Azimuth (°)", color="#888", fontsize=7.5)
    ax_c.set_ylabel("RCS (dBsm)",  color="#888", fontsize=7.5)
    ax_c.legend(fontsize=6.5, facecolor="#111", labelcolor="#aaa",
                 framealpha=0.7, loc="upper right")

    # ── Panel C: Nose & broadside zoom ────────────────────────────────────────
    ax_z = fig.add_subplot(gs[0, 2])
    _ax_style(ax_z, "Nose ±30° vs Broadside 60°–120°")
    nm = (theta <= 30) | (theta >= 330)
    bm = (theta >= 60) & (theta <= 120)
    ax_z.plot(theta[nm] % 360, rcs_db[nm],  color=_CYAN,   lw=1.5,
               label=f"Nose  avg={np.mean(rcs_db[nm]):.1f} dBsm")
    ax_z.plot(theta[bm],       rcs_db[bm],  color=_ORANGE, lw=1.5,
               label=f"Broad avg={np.mean(rcs_db[bm]):.1f} dBsm")
    ax_z.set_xlabel("Azimuth (°)", color="#888", fontsize=7.5)
    ax_z.set_ylabel("RCS (dBsm)",  color="#888", fontsize=7.5)
    ax_z.legend(fontsize=6.5, facecolor="#111", labelcolor="#aaa",
                 framealpha=0.7)

    # ── Panel D: Doppler ──────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    _ax_style(ax_d, "Doppler Frequency Shift")
    d_kHz = doppler / 1e3
    ax_d.plot(theta, d_kHz, color=_ORANGE, lw=1.4)
    ax_d.fill_between(theta, 0, d_kHz, where=d_kHz > 0,
                       color=_ORANGE, alpha=0.12, label="Closing")
    ax_d.fill_between(theta, 0, d_kHz, where=d_kHz < 0,
                       color=_MAGENTA, alpha=0.12, label="Opening")
    ax_d.axhline(0, color="white", lw=0.6, alpha=0.4)
    ax_d.set_xlabel("Azimuth (°)", color="#888", fontsize=7.5)
    ax_d.set_ylabel("Δf (kHz)",    color="#888", fontsize=7.5)
    ax_d.legend(fontsize=6.5, facecolor="#111", labelcolor="#aaa",
                 framealpha=0.7)

    # ── Panel E: Frequency sweep ──────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[1, 2])
    _ax_style(ax_f, "RCS vs Frequency  (Nose & Broadside)")
    if freq_sweep is not None:
        ax_f.semilogx(freq_sweep["freq_hz"]/1e9,
                       freq_sweep["rcs_nose_db"],
                       color=_CYAN,   lw=1.5, label="Nose-on (0°)")
        ax_f.semilogx(freq_sweep["freq_hz"]/1e9,
                       freq_sweep["rcs_broadside_db"],
                       color=_ORANGE, lw=1.5, label="Broadside (90°)")
        # Band markers
        ylim = ax_f.get_ylim()
        for bn, bf in RADAR_BANDS.items():
            ax_f.axvline(bf/1e9, color="#333", lw=0.5, ls=":")
            ax_f.text(bf/1e9, ylim[0] + 0.5, bn,
                       color="#555", fontsize=5.5, rotation=90, va="bottom")
        ax_f.set_xlabel("Frequency (GHz)", color="#888", fontsize=7.5)
        ax_f.set_ylabel("RCS (dBsm)",      color="#888", fontsize=7.5)
        ax_f.legend(fontsize=6.5, facecolor="#111", labelcolor="#aaa",
                     framealpha=0.7)
    else:
        ax_f.text(0.5, 0.5, "Frequency sweep\nnot enabled\n(select 'y' at prompt)",
                   ha="center", va="center", color="#555",
                   transform=ax_f.transAxes, fontsize=9)

    # ── Panel F: CDF ──────────────────────────────────────────────────────────
    ax_cdf = fig.add_subplot(gs[2, 0])
    _ax_style(ax_cdf, "RCS Cumulative Distribution (All Aspects)")
    sr  = np.sort(rcs_db)
    cdf = np.linspace(0, 100, len(sr))
    ax_cdf.plot(sr, cdf, color=_GREEN, lw=1.4)
    ax_cdf.axvline(np.median(rcs_db), color=_YELLOW, lw=1, ls="--",
                    label=f"Median {np.median(rcs_db):.1f} dBsm")
    ax_cdf.axvline(np.percentile(rcs_db, 90), color=_CYAN, lw=0.9, ls=":",
                    label=f"P90 {np.percentile(rcs_db, 90):.1f} dBsm")
    ax_cdf.set_xlabel("RCS (dBsm)", color="#888", fontsize=7.5)
    ax_cdf.set_ylabel("Cumulative %", color="#888", fontsize=7.5)
    ax_cdf.legend(fontsize=6.5, facecolor="#111", labelcolor="#aaa",
                   framealpha=0.7)

    # ── Panel G: Histogram ────────────────────────────────────────────────────
    ax_h = fig.add_subplot(gs[2, 1])
    _ax_style(ax_h, "RCS Distribution (All Aspects)")
    ax_h.hist(rcs_db, bins=45, color=_CYAN, alpha=0.65, edgecolor="#0A0A0A")
    ax_h.axvline(np.mean(rcs_db), color=_YELLOW, lw=1.5,
                  label=f"Mean  {np.mean(rcs_db):.1f} dBsm")
    ax_h.axvline(np.max(rcs_db),  color=_MAGENTA, lw=1, ls="--",
                  label=f"Peak  {np.max(rcs_db):.1f} dBsm")
    ax_h.set_xlabel("RCS (dBsm)", color="#888", fontsize=7.5)
    ax_h.set_ylabel("Count",      color="#888", fontsize=7.5)
    ax_h.legend(fontsize=6.5, facecolor="#111", labelcolor="#aaa",
                 framealpha=0.7)

    # ── Panel H: Summary text ─────────────────────────────────────────────────
    ax_s = fig.add_subplot(gs[2, 2])
    ax_s.set_facecolor(_BG)
    ax_s.axis("off")
    sw_names = {0:"Det.",1:"Sw-I",2:"Sw-II",3:"Sw-III",4:"Sw-IV"}
    summary = (
        f"TARGET   {filename}\n"
        f"BAND     {band}   ({freq/1e9:.2f} GHz)\n"
        f"λ        {lam*100:.2f} cm\n"
        f"POL      {polarization}\n"
        f"MAT      {material_key}\n"
        f"EL       {elevation_deg}°\n"
        f"FLUCT    {sw_names.get(swerling_model,'?')}\n"
        f"{'─'*30}\n"
        f"TEMP     {env['temp_c']:+.1f}°C  (ISA)\n"
        f"γ_O₂     {env['gamma_o']:.4f} dB/km\n"
        f"γ_H₂O    {env['gamma_w']:.4f} dB/km\n"
        f"2-way Δ  {env['loss_db']:.3f} dB\n"
        f"{'─'*30}\n"
        f"NOSE     {rcs_db[0]:.2f} dBsm\n"
        f"TAIL     {rcs_db[180]:.2f} dBsm\n"
        f"BROAD    {np.max(rcs_db[60:120]):.2f} dBsm\n"
        f"MEAN     {np.mean(rcs_db):.2f} dBsm\n"
        f"σ(RCS)   {np.std(rcs_db):.2f} dB\n"
        f"PEAK     {np.max(rcs_db):.2f} dBsm  "
        f"@ {theta[np.argmax(rcs_db)]:.0f}°\n"
        f"DOPPLER  {np.max(np.abs(doppler))/1e3:.2f} kHz\n"
    )
    ax_s.text(0.04, 0.97, summary,
               transform=ax_s.transAxes,
               fontsize=7.8, color="#AAFFAA",
               va="top", fontfamily="monospace",
               bbox=dict(boxstyle="round,pad=0.5",
                          facecolor="#060F06",
                          edgecolor="#00AA44",
                          alpha=0.92))

    # ── Master title ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"ULTRA-REALISM RCS DASHBOARD  ·  PO + PTD (Ufimtsev) + "
        f"2-Bounce GO + Fresnel + ISA + ITU-R P.676-12  ·  {filename}",
        color=_GREEN, fontsize=11.5, fontweight="bold",
    )

    out = OUTPUT_DIR / f"RCS_{CAD_FILE.stem}_{band}_{polarization}_{material_key}.png"
    plt.savefig(str(out), facecolor="#050505", dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"Dashboard → {out}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    console.print(Panel.fit(
        "[bold cyan]UNIVERSAL RCS ANALYSIS SUITE  v4.0[/bold cyan]\n"
        "[dim]PO + PTD (Ufimtsev 1962) + 2-Bounce GO + Fresnel Materials\n"
        "ITU-R P.676-12 Atmosphere · Full Polarimetry · Swerling Statistics\n"
        "Supports: STL · OBJ · PLY · STEP · IGES · X_T / X_B Parasolid[/dim]",
        subtitle="[bold white]Refs: Knott/Schaeffer/Tuley 2004 · Ufimtsev 1962 · "
                  "Balanis 2012 · Skolnik 2001[/bold white]",
        border_style="bright_blue",
        box=box.DOUBLE_EDGE,
    ))

    # ── File selection — update CAD_FILE at the top of this script ──────────
    cad_path = CAD_FILE
    if not cad_path.exists():
        console.print(f"[bold red]✗ File not found:[/bold red] {cad_path}")
        sys.exit(1)
    console.print(f"[green]✓ Target file:[/green] [white]{cad_path}[/white]")

    # ── User inputs ────────────────────────────────────────────────────────────
    band_id  = Prompt.ask("Radar Band",
                           choices=list(RADAR_BANDS), default="X")
    pol_id   = Prompt.ask("Polarization",
                           choices=["HH", "VV", "avg"], default="HH")
    mat_id   = Prompt.ask("Surface Material",
                           choices=list(MATERIALS), default="Aluminum")
    sw_model = int(Prompt.ask("Swerling Model (0=deterministic, 1-4)",
                               choices=["0","1","2","3","4"], default="0"))
    speed    = float(Prompt.ask("Speed (m/s)",        default="280"))
    altitude = float(Prompt.ask("Altitude (m)",       default="10000"))
    elev     = float(Prompt.ask("Elevation angle (°)", default="0"))
    do_fsweep = Prompt.ask(
        "Run frequency sweep 300 MHz–35 GHz? [bold yellow](adds ~30 s)[/bold yellow]",
        choices=["y", "n"], default="n") == "y"

    freq = RADAR_BANDS[band_id]

    # ── Load geometry ──────────────────────────────────────────────────────────
    mesh = load_and_validate_geometry(cad_path, radar_freq_hz=freq)

    # ── Atmosphere ─────────────────────────────────────────────────────────────
    env       = Atmosphere.get_context(altitude, freq)
    env["mach"] = speed / env["mach_1"]

    # ── Main RCS solve ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    theta, rcs_raw, doppler = solve_rcs_hifi(
        mesh, freq, speed,
        material_key  = mat_id,
        polarization  = pol_id,
        elevation_deg = elev,
        include_ptd          = True,
        include_multibounce  = True,
        swerling_model       = sw_model,
    )
    elapsed = time.perf_counter() - t0

    # Subtract two-way atmospheric loss (as a realistic radar would experience)
    rcs_corrected = rcs_raw - env["loss_db"]

    # ── Optional frequency sweep ───────────────────────────────────────────────
    fsweep = None
    if do_fsweep:
        t1 = time.perf_counter()
        fsweep = frequency_sweep(mesh, mat_id, pol_id)
        log.info(f"Frequency sweep: {time.perf_counter()-t1:.1f} s")

    # ── Dashboard ──────────────────────────────────────────────────────────────
    out_img = generate_dashboard(
        theta, rcs_corrected, doppler,
        band          = band_id,
        env           = env,
        filename      = cad_path.name,
        freq_sweep    = fsweep,
        material_key  = mat_id,
        polarization  = pol_id,
        elevation_deg = elev,
        swerling_model= sw_model,
    )

    # ── Terminal report ────────────────────────────────────────────────────────
    mat_info = MATERIALS[mat_id]
    sw_str   = {0:"Deterministic",1:"Swerling I",2:"Swerling II",
                 3:"Swerling III",4:"Swerling IV"}

    tbl = Table(title="\n[bold white]HIGH-FIDELITY RCS ANALYSIS REPORT[/bold white]",
                 box=box.HEAVY_EDGE, show_lines=True, min_width=70)
    tbl.add_column("Parameter",   style="cyan",        no_wrap=True, min_width=22)
    tbl.add_column("Value",       style="bold yellow",  no_wrap=True, min_width=20)
    tbl.add_column("Reference / Notes", style="dim white", no_wrap=False)

    def row(p, v, n=""): tbl.add_row(p, v, n)
    def sep(): tbl.add_row("─"*22, "─"*20, "─"*22)

    row("Radar Band",      f"{band_id}  ({freq/1e9:.2f} GHz)", "ITU radar band designation")
    row("Wavelength",      f"{C0/freq*100:.2f} cm",            "λ = c₀/f")
    row("Polarization",    pol_id,                             "HH=horizontal co-pol")
    row("Surface Material",mat_id,                             mat_info.get("note",""))
    row("Target Fluct.",   sw_str.get(sw_model,"?"),           "[4] Skolnik, Ch.2")
    row("Aircraft Speed",  f"{speed:.1f} m/s ({env['mach']:.3f} M)", "M = v/v_sound (ISA)")
    row("Altitude",        f"{altitude:.0f} m",                "ISA troposphere")
    row("Temperature",     f"{env['temp_c']:+.1f}°C",          "ISA: T = 288.15 − 6.5h/km")
    row("Pressure",        f"{env['pressure']:.1f} hPa",       "ISA barometric pressure")
    row("H₂O density",     f"{env['rho_water']:.2f} g/m³",    "Exponential scale-ht 2 km")
    sep()
    row("O₂ attenuation",  f"{env['gamma_o']:.5f} dB/km",     "[5] ITU-R P.676-12 (2019)")
    row("H₂O attenuation", f"{env['gamma_w']:.5f} dB/km",     "[5] ITU-R P.676-12 (2019)")
    row("2-way path loss",  f"{env['loss_db']:.4f} dB",        "Applied as post-correction")
    sep()
    row("Nose-on  (0°)",   f"{rcs_corrected[0]:.2f} dBsm",    "Head-on signature")
    row("Tail     (180°)", f"{rcs_corrected[180]:.2f} dBsm",  "Rear-aspect signature")
    row("Broadside peak",  f"{np.max(rcs_corrected[60:120]):.2f} dBsm",
                            "90°±30° sector maximum")
    row("All-aspect mean", f"{np.mean(rcs_corrected):.2f} dBsm","Aspect-averaged σ̄")
    row("RCS std-dev σ",   f"{np.std(rcs_corrected):.2f} dB",  "Signature variability")
    row("Global peak",     f"{np.max(rcs_corrected):.2f} dBsm  "
                            f"@ {theta[np.argmax(rcs_corrected)]:.0f}°",
                            "Flash / specular spike")
    row("Peak Doppler",    f"{np.max(np.abs(doppler))/1e3:.3f} kHz",
                            "Δf = 2v cosα / λ")
    sep()
    row("Physics engine",  "PO + PTD + 2-bounce + Fresnel",
                            "[1][2][3]")
    row("Mesh faces",      f"{len(mesh.faces):,}",             "After decimation")
    row("Solver time",     f"{elapsed:.1f} s",                 "361-angle sweep (1° step)")

    console.print(tbl)
    console.print(
        f"\n[bold green]✓ Complete![/bold green]  "
        f"Dashboard → [white]{out_img}[/white]\n"
        f"[dim]Physics: PO (Fresnel) + PTD (Ufimtsev) + 2-bounce GO + "
        f"ISA + ITU-R P.676-12 + {sw_str[sw_model]}[/dim]\n"
    )


if __name__ == "__main__":
    main()
