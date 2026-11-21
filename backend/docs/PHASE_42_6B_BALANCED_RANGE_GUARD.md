# SkyGuard FUSION â€” Phase 42.6B: Evidence-gated camera optical-axis depth (software)

## Purpose

This additive module is **not** a replacement for the existing `app/localization` implementation. It provides a conservative interface: when evidence for a quantitative range is missing or invalid, return `status=unknown` and **no numeric distance**. In phase 45 it must be assessed against independently measured physical distances and flight poses. Until then, no physical accuracy or operational validity is claimed.

## Estimation contract

Inputs: calibrated focal length in **pixels** (with measurement interval and source), independently verified **effective apparent target dimension** (with orientation/pose accounted for; interval and source), and corresponding width or height in pixels (with interval and source), associated frame and track IDs.

A pinhole approximation estimates **camera optical-axis depth** in metres:

`Z_est = focal_px * apparent_target_size_m / measured_image_size_px`

For independent declared bounded inputs (not statistical confidence intervals):

`Z_lower = focal_lower * target_size_lower / image_size_upper`

`Z_upper = focal_upper * target_size_upper / image_size_lower`

This is an approximation and it is only meaningful when the apparent dimension and image-axis match for the observed target pose. The `calibrated_camera` and `independently_sized_target` flags are assertions by the calling integration and must be grounded in actual recorded evidence. Changing a flag is not a substitute for a measurement.

## Hard safety boundaries

- Pixel center and bearing alone **cannot determine range**.
- An uncalibrated camera, unknown drone size/pose, missing or invalid measurement, or numerical instability returns **UNKNOWN**.
- No facility-relative range or latitude/longitude is ever produced by this module. The facility's offset, altitude, camera orientation and site reference system require independent evidence and explicit transformation by an appropriate component.
- Bounds are conditional on the declared measurement intervals; they are **not** 95% confidence intervals and must not be presented that way.
- One-pixel targets are highly unstable. They are not evidence of accurate far-range estimation.
- No physical camera/drone test is performed here (reserved for Phase 45).

## Implementation status

This phase creates a new pure module and targeted tests only, without changing the production geo API or tracker. It requires a later integration task to map real calibration/provenance and track observations into this contract, and to add UNKNOWN handling in the live UI. No demo coordinates, range measurements or fake GPS output.

## Test steps

The accompanying PowerShell script creates files only when none exists. It runs targeted pytest, then (unless `-SkipFullRegression`) backend regression tests. All output reports are written outside the repository in Downloads. No Git add, commit, push, merge or cherry-pick are executed. On failed tests, only the files created by this script are removed; pre-existing files are left intact.
