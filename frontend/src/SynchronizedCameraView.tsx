import { useState } from 'react'
import type { LiveSynchronizedFrame } from './api/liveCamera'
import TrajectoryOverlay from './TrajectoryOverlay'

interface Props {
  frame: LiveSynchronizedFrame | null
}

export default function SynchronizedCameraView({ frame }: Props) {
  const [dimensions, setDimensions] = useState<{
    width: number
    height: number
    frameIndex: number
  } | null>(null)

  if (!frame) {
    return (
      <div className="camera-offline">
        <p>Waiting for synchronized camera frame...</p>
      </div>
    )
  }

  const imageSource = `data:image/jpeg;base64,${frame.image_base64}`

  const validDimensions =
    dimensions !== null &&
    dimensions.frameIndex === frame.frame_index &&
    dimensions.width > 0 &&
    dimensions.height > 0

  const confirmedIds = new Set(
    frame.tracks
      .filter(
        track =>
          track.state === 'confirmed' &&
          track.frame_index === frame.frame_index,
      )
      .map(track => track.track_id),
  )

  const trajectories = frame.trajectories.filter(trajectory => {
    const points = trajectory.points

    return (
      confirmedIds.has(trajectory.track_id) &&
      Array.isArray(points) &&
      points.length >= 2 &&
      points[points.length - 1].frame_index === frame.frame_index
    )
  })

  return (
    <div className="synchronized-camera-container">
      <div
        className="synchronized-camera-content"
        style={
          validDimensions
            ? {
                aspectRatio: `${dimensions.width} / ${dimensions.height}`,
                width: `min(100%, calc(100cqh * ${dimensions.width} / ${dimensions.height}))`,
              }
            : undefined
        }
      >
        <img
          key={frame.frame_index}
          src={imageSource}
          alt="Synchronized SkyGuard camera frame"
          className="synchronized-camera-image"
          onLoad={event => {
            const image = event.currentTarget

            if (image.naturalWidth > 0 && image.naturalHeight > 0) {
              setDimensions({
                width: image.naturalWidth,
                height: image.naturalHeight,
                frameIndex: frame.frame_index,
              })
            }
          }}
        />

        {validDimensions && (
          <TrajectoryOverlay
            trajectories={trajectories}
            imageWidth={dimensions.width}
            imageHeight={dimensions.height}
          />
        )}
      </div>
    </div>
  )
}
