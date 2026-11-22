import type {Benchmark, Camera} from './types'
import {validCoordinate} from './logic'

export type Coordinate = [number, number]

const earthRadius = 6371008.8
const toRadians = Math.PI / 180
const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null

export function offsetCoordinate([lon, lat]: Coordinate, bearing: number, meters: number): Coordinate {
  const a = lat * toRadians
  const b = bearing * toRadians
  const distance = meters / earthRadius
  const endLat = Math.asin(Math.sin(a) * Math.cos(distance) + Math.cos(a) * Math.sin(distance) * Math.cos(b))
  const endLon = lon + Math.atan2(Math.sin(b) * Math.sin(distance) * Math.cos(a),
    Math.cos(distance) - Math.sin(a) * Math.sin(endLat)) / toRadians
  return [endLon, endLat / toRadians]
}

export function scaleRings(center: Coordinate, zoom: number) {
  const metersPerPixel = 2 * Math.PI * earthRadius * Math.cos(center[1] * toRadians) / (512 * 2 ** zoom)
  const target = Math.min(200000, Math.max(1, metersPerPixel * 74))
  const decade = 10 ** Math.floor(Math.log10(target))
  const step = ([1, 2, 5, 10].find(n => n * decade >= target) ?? 10) * decade
  return [1, 2, 3].map(index => {
    const meters = step * index
    return { meters, label: offsetCoordinate(center, 80, meters),
      line: Array.from({length: 73}, (_, i) => offsetCoordinate(center, i * 5, meters)) }
  })
}

export function orientationSector(center: Coordinate, heading: number, fieldOfView: number, meters: number): Coordinate[] {
  return [center, ...Array.from({length: 33}, (_, i) =>
    offsetCoordinate(center, heading - fieldOfView / 2 + i * fieldOfView / 32, meters)), center]
}

type Trial = { coordinate: Coordinate; horizontalMeters: number; passed: boolean; bearing: number }

export function verifiedCameraCoverage(benchmarks: Benchmark[], camera: Camera) {
  for (const benchmark of benchmarks) {
    const record = benchmark.metrics?.camera_coverage_v1
    if (benchmark.validation_level !== 'PHYSICAL_VERIFIED' || !isRecord(record) ||
        record.camera_id !== camera.id || record.calibration_ref !== camera.calibration_ref ||
        record.latitude !== camera.latitude || record.longitude !== camera.longitude ||
        typeof record.evidence_ref !== 'string' || !record.evidence_ref.trim()) continue
    const trials: Trial[] = []
    if (Array.isArray(record.trials)) {
      for (const trial of record.trials) {
        if (!isRecord(trial) || !validCoordinate(trial.latitude, trial.longitude) ||
            !isNumber(trial.horizontal_m) || trial.horizontal_m <= 0 ||
            !isNumber(trial.bearing_deg) || trial.bearing_deg < 0 || trial.bearing_deg >= 360 ||
            typeof trial.passed !== 'boolean') continue
        trials.push({ coordinate: [trial.longitude as number, trial.latitude], horizontalMeters: trial.horizontal_m,
          passed: trial.passed, bearing: trial.bearing_deg })
      }
    }
    const qualified = trials.filter(trial => trial.passed)
    const direction = qualified[0]?.bearing
    const multipleDirections = direction !== undefined && qualified.some(trial =>
      Math.abs(((trial.bearing - direction + 540) % 360) - 180) >= 5)
    const suppliedArea = record.verified_area
    const area: Coordinate[] = multipleDirections && Array.isArray(suppliedArea) && suppliedArea.length >= 3 &&
      suppliedArea.every(p => Array.isArray(p) && p.length === 2 && validCoordinate(p[1], p[0]))
      ? suppliedArea as Coordinate[] : []
    const orientation = record.fixed_camera === true && isNumber(record.heading_deg) &&
      record.heading_deg >= 0 && record.heading_deg < 360 && isNumber(record.horizontal_fov_deg) &&
      record.horizontal_fov_deg > 0 && record.horizontal_fov_deg <= 180
      ? { heading: record.heading_deg, fieldOfView: record.horizontal_fov_deg } : null
    const source = typeof record.position_source === 'string' && record.position_source.trim()
      ? record.position_source : null
    const uncertainty = isNumber(record.position_uncertainty_m) && record.position_uncertainty_m >= 0
      ? record.position_uncertainty_m : null
    return { trials, area, orientation, source, uncertainty, evidenceRef: record.evidence_ref,
      maximumVerified: qualified.length ? Math.max(...qualified.map(trial => trial.horizontalMeters)) : null,
      multipleDirections }
  }
  return null
}
