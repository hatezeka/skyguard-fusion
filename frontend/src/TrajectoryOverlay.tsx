import type { LiveTrajectory } from './api/liveCamera'

interface TrajectoryOverlayProps {
  trajectories: LiveTrajectory[]
  imageWidth: number
  imageHeight: number
  maxTracks?: number
}

export default function TrajectoryOverlay({
  trajectories,
  imageWidth,
  imageHeight,
  maxTracks = 20,
}: TrajectoryOverlayProps) {
  if (
    !Number.isFinite(imageWidth) ||
    !Number.isFinite(imageHeight) ||
    imageWidth <= 0 ||
    imageHeight <= 0
  ) {
    return null
  }

  const visible = trajectories.slice(0, maxTracks)

  return (
    <svg
      className="trajectory-overlay"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="xMidYMid meet"
      aria-label="Estimated image-space trajectories"
      role="img"
    >
      {visible.map(trajectory => {
        const validPoints = trajectory.points.filter(
          point =>
            Number.isFinite(point.x_px) &&
            Number.isFinite(point.y_px) &&
            point.x_px >= 0 &&
            point.y_px >= 0 &&
            point.x_px <= imageWidth &&
            point.y_px <= imageHeight,
        )

        if (validPoints.length < 2) {
          return null
        }

        const points = validPoints
          .map(point => `${point.x_px},${point.y_px}`)
          .join(' ')

        const last = validPoints[validPoints.length - 1]

        return (
          <g key={trajectory.track_id}>
            <polyline
              points={points}
              fill="none"
              stroke={
                trajectory.object_class === 'drone'
                  ? '#ff6464'
                  : trajectory.object_class === 'bird'
                    ? '#6ec8ff'
                    : '#f5b74f'
              }
              strokeWidth="2.5"
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity="0.85"
            />
            <circle
              cx={last.x_px}
              cy={last.y_px}
              r="4"
              fill="#ffffff"
              stroke="#101a15"
              strokeWidth="1.5"
            />
          </g>
        )
      })}
    </svg>
  )
}
