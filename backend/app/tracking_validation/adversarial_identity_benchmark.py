"""Deterministic adversarial identity benchmark for temporal tracking."""

from dataclasses import dataclass
from random import Random

from app.detection.temporal_hypotheses import TemporalHypothesisManager


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    correct: int
    wrong: int
    ambiguous: int
    missing: int
    excluded_overlap: int
    frames: int

    @property
    def coverage(self):
        total = self.correct + self.wrong + self.ambiguous + self.missing
        return (self.correct + self.wrong) / total if total else 0.0

    @property
    def confirmed_error_rate(self):
        total = self.correct + self.wrong
        return self.wrong / total if total else 0.0


def run_scenario(name, seed=42, frames=21):
    rng = Random(seed)

    initial = [(20.0, 50.0), (120.0, 50.0)]
    manager = TemporalHypothesisManager(initial)

    correct = wrong = ambiguous = missing = excluded = 0

    for t in range(frames):
        truth = [
            (20.0 + 5.0 * t, 50.0),
            (120.0 - 5.0 * t, 50.0),
        ]

        if name == "maneuver" and t >= 11:
            truth = [
                (75.0 - 3.0 * (t - 11), 50.0),
                (65.0 + 3.0 * (t - 11), 50.0),
            ]

        observations = []

        for identity, (x, y) in enumerate(truth):
            if name == "missed" and identity == 0 and t in (8, 9, 10):
                continue

            if name == "noise":
                x += rng.gauss(0, 2.0)
                y += rng.gauss(0, 2.0)

            observations.append(((x, y), identity))

        if name == "clutter" and t in (5, 10, 15):
            observations.append(((rng.uniform(0, 140), 50.0), None))

        if name == "shuffle":
            rng.shuffle(observations)

        detections = [point for point, _ in observations]

        decision = manager.update(t, detections)

        # Exact coincident observations cannot be distinguished by position.
        overlap = (
            len(observations) >= 2
            and any(
                observations[i][0] == observations[j][0]
                for i in range(len(observations))
                for j in range(i + 1, len(observations))
                if observations[i][1] is not None
                and observations[j][1] is not None
            )
        )

        if overlap:
            excluded += 2
            continue

        assigned_tracks = set()

        for track_index, detection_index in decision.assignments:
            assigned_tracks.add(track_index)
            actual_id = observations[detection_index][1]

            if actual_id == track_index:
                correct += 1
            else:
                wrong += 1

        ambiguous_tracks = set(decision.ambiguous_tracks)
        ambiguous += len(ambiguous_tracks)

        # Any remaining track without a confirmed or ambiguous assignment
        # is counted separately, rather than silently ignored.
        for track_index in range(len(initial)):
            if (
                track_index not in assigned_tracks
                and track_index not in ambiguous_tracks
            ):
                missing += 1

    return BenchmarkResult(
        scenario=name,
        correct=correct,
        wrong=wrong,
        ambiguous=ambiguous,
        missing=missing,
        excluded_overlap=excluded,
        frames=frames,
    )


def main():
    scenarios = (
        "baseline",
        "noise",
        "shuffle",
        "missed",
        "clutter",
        "maneuver",
    )

    print("SKYGUARD ADVERSARIAL IDENTITY BENCHMARK")
    print("=" * 76)

    for scenario in scenarios:
        result = run_scenario(scenario)

        print(
            f"{scenario:10} "
            f"correct={result.correct:3} "
            f"wrong={result.wrong:3} "
            f"ambiguous={result.ambiguous:3} "
            f"missing={result.missing:3} "
            f"excluded={result.excluded_overlap:2} "
            f"coverage={result.coverage:6.1%} "
            f"error={result.confirmed_error_rate:6.1%}"
        )


if __name__ == "__main__":
    main()
