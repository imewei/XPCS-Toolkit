# GUI Review – XPCS Viewer

## Evidence
- Offscreen harness: `QT_QPA_PLATFORM=offscreen python3 scripts/offscreen_snap.py`
- Outputs: `screenshots/offscreen/00_initial.png` … `09_Metadata.png`; status log `screenshots/offscreen/status.txt` (load_path succeeded).
- Harness script: `scripts/offscreen_snap.py`.

**Addendum (latest quick wins + smoke)**
- Applied quick wins: clearer labels/tooltips/accessibility on path/browse/reload/filter/add/remove/target controls; workflow-first tab order; empty-state guardrails prevent plotting with no targets; progress now surfaces by default in status bar.
- Offscreen smoke (after changes): `python tests/gui_interactive/offscreen_snap.py --output tests/artifacts/offscreen_snap.png` – pass. Warnings: Qt “propagateSizeHints” plugin warning persists (benign); no data loaded (expected) but now guarded.
- Next actions: hook snapshot into CI with golden comparison; add menu/toolbar; padding/spacing polish; sample/recent picker; broaden shortcuts.

**Addendum 2 (CI + UI polish)**
- CI: added golden snapshot test `tests/gui_interactive/test_offscreen_snapshot.py` with baseline `tests/gui_interactive/goldens/offscreen_snap.png`; workflow now runs it headlessly.
- UI: added File/View/Help menu + toolbar (Open, Reload, Progress, View Logs), padding/spacing on central layout, startup prompt offering sample/browse, and new shortcuts (Ctrl+O, Ctrl+R, Ctrl+Shift+A, Ctrl+P, Ctrl+L, F1).
- Snapshot refreshed after UI changes; current golden matches `tests/artifacts/offscreen_snap.png` (2025-12-09 run).

How to rerun:
1) `QT_QPA_PLATFORM=offscreen python3 scripts/offscreen_snap.py`
2) Screenshots land in `screenshots/offscreen/*.png`; run status in `screenshots/offscreen/status.txt`.
3) Remove/refresh the folder before comparing to goldens.

## Key Findings
- **IA / Navigation**: 10 crowded tabs (SAXS-2D, SAXS-1D, Stability, Intensity-Time, g2, Diffusion, Two Time, QMap, Average, Metadata); no menu/toolbar; empty states unclear—tabs stay enabled when incompatible.
- **Labeling / Accessibility / Tab Order**: Ambiguous labels and generic widget names; 103 tabstops in source order; missing accessible names/descriptions; sparse tooltips; few shortcuts.
- **Layout / Theming / Icons**: Tight grid spacing, mixed button heights, fixed 1024×800 default; default Qt palette, no dark/contrast handling; single giraffe app icon, no action icons; screenshots show cramped controls.
- **Performance / Responsiveness**: Async infra exists but heavy paths still on UI thread; Diffusion tab does plotting with no data and emits warnings; progress/cancel not surfaced consistently; thread pool fixed (20) without backpressure; no startup feedback.
- **Error Handling**: Errors mostly status-bar/log-only; dialogs inconsistent and non-actionable; no retry; empty-state guidance missing.
- **First-run / Discovery**: Starts in CWD; no sample-data or recent-files prompt; users must locate data manually.

## Prioritized Improvement Plan
**Quick wins (1–2 days)**
- Rename key actions; add tooltips + `accessibleName/Description`; clarify export/plot buttons.
- Reorder tabstops per workflow; add Open/Export/tab shortcuts; remove unused stops.
- Inline empty-state banners; skip plotting when no targets; add “Load sample data” CTA.
- Always show progress dialog for long ops; expose Cancel everywhere supported.
- Add padding/margins; normalize button heights; fix cramped grids noted in screenshots.
- Startup polish: splash/progress text; remove duplicate cleanup log; show sample/recent chooser.

**Medium term (1–2 sprints)**
- Add menu/toolbar for core actions; consider grouping tabs into left nav/subtabs by workflow.
- Define light/dark palette with contrast; add HiDPI app/action icons.
- Make async default for load/plot/export; show queue/ETA in progress manager; add backpressure on thread pool.
- Standard error dialogs with cause/remedy/retry/log-link; replace status-only failures.
- Adaptive window/HiDPI sizing; avoid matplotlib “tight layout” warnings.

**Long term**
- Onboarding/wizard (Open → Configure → Run → Review → Export) to reduce cognitive load.
- Localization and deeper accessibility (focus rings on plots, contrast checks).
- Perf harness with guardrails for UI-thread stalls.

## Validation / Testing Recommendations
- Keep `scripts/offscreen_snap.py` in CI: generate per-tab PNGs and compare to goldens.
- Add tab-order and shortcut tests; assert primary actions reachable quickly.
- Empty-state tests: no targets ⇒ no plotting; banner shown.
- Perf smoke: timed load/plot; assert main-thread stalls <200 ms.
- Error-dialog tests: simulate bad/missing data and permission errors; verify actionable text + log link.

## Prioritized Backlog (finding → impact → priority/effort)
- Ambiguous labels/tooltips/accessibility missing → User confusion & AT unusable → P1 / S
- Tab order in source order (103 stops) → Keyboard navigation painful → P1 / S
- No empty-state guidance; costly plotting even with no targets → Wasted work, unclear state → P1 / S
- Progress/cancel not consistent; heavy ops on UI thread → Freeze risk on large data → P1 / M
- No menu/toolbar; actions buried in tabs → Discoverability low → P2 / M
- Default palette, no icons, cramped grids → Visual fatigue, low clarity → P2 / M
- Startup lacks sample/recent picker → Poor first-run success → P2 / S
- Error UX status-bar only → Users miss failures, no retry → P2 / M
- Fixed 1024×800, HiDPI not handled → Poor scaling on modern displays → P3 / M
- No golden-image/tab-order tests → Regressions likely → P3 / M
- No onboarding/wizard → High cognitive load → P3 / L
- Localization/contrast hardening absent → Accessibility gaps → P3 / L

## Immediate Next Actions (top quick wins)
1) Rename ambiguous buttons/labels; add tooltips + `accessibleName/Description`; commit in `xpcs.ui`.
2) Reorder tabstops per tab workflow and add keyboard shortcuts for Open/Export/tab switching.
3) Add empty-state banners and skip plotting when no target files; disable incompatible tabs.
4) Ensure progress dialog + Cancel surface for long ops; default heavy paths to async.
5) Add startup sample-data/recent-files chooser and modest padding/spacing pass to reduce crowding.
