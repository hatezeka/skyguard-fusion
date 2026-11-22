from skyguard_fix.core import Detection, apply_bird_evidence


def test_one_matching_independent_bird_box_corrects_drone():
    original = Detection("DRONE", .74, (0, 0, 100, 100), "primary")
    bird = Detection("BIRD", .83, (2, 2, 99, 99), "verifier")
    corrected, = apply_bird_evidence([original], [bird])
    assert corrected.label == "BIRD"
    assert corrected.vote_evidence == (("DRONE", .74), ("BIRD", .83))


def test_verifier_requires_strong_unambiguous_spatial_match():
    original = Detection("DRONE", .74, (0, 0, 100, 100), "primary")
    weak = Detection("BIRD", .64, (0, 0, 100, 100), "verifier")
    distant = Detection("BIRD", .92, (110, 0, 200, 100), "verifier")
    assert apply_bird_evidence([original], [weak, distant]) == [original]
    assert apply_bird_evidence([original], []) == [original]


def test_unknown_cannot_be_relabelled_by_bird_verifier():
    unknown = Detection("UNKNOWN", .74, (0, 0, 100, 100), "primary")
    bird = Detection("BIRD", .83, (2, 2, 99, 99), "verifier")
    assert apply_bird_evidence([unknown], [bird]) == [unknown]
    assert unknown.track_id is None
