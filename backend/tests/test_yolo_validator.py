import pytest

from app.datasets.yolo_validator import validate_dataset


def make_dataset(root):
    for split in ("train", "val", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)

        (root / split / "images" / "sample.jpg").write_bytes(
            split.encode()
        )

        (root / split / "labels" / "sample.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )


def test_valid_dataset(tmp_path):
    make_dataset(tmp_path)

    report = validate_dataset(tmp_path)

    assert report.valid
    assert report.images == 3
    assert report.labels == 3
    assert report.annotations == 3


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_dataset(tmp_path / "missing")


def test_invalid_class(tmp_path):
    make_dataset(tmp_path)

    label = tmp_path / "train" / "labels" / "sample.txt"
    label.write_text("9 0.5 0.5 0.2 0.2\n")

    report = validate_dataset(tmp_path)

    assert not report.valid
    assert any("Invalid class" in error for error in report.errors)


def test_invalid_box(tmp_path):
    make_dataset(tmp_path)

    label = tmp_path / "train" / "labels" / "sample.txt"
    label.write_text("0 0.9 0.5 0.5 0.2\n")

    report = validate_dataset(tmp_path)

    assert not report.valid
    assert any("Invalid bounding box" in error for error in report.errors)


def test_cross_split_duplicate(tmp_path):
    make_dataset(tmp_path)

    source = tmp_path / "train" / "images" / "sample.jpg"
    target = tmp_path / "val" / "images" / "sample.jpg"

    target.write_bytes(source.read_bytes())

    report = validate_dataset(tmp_path)

    assert not report.valid
    assert report.duplicate_groups >= 1


def test_missing_label(tmp_path):
    make_dataset(tmp_path)

    label = tmp_path / "test" / "labels" / "sample.txt"
    label.unlink()

    report = validate_dataset(tmp_path)

    assert not report.valid
    assert any("Missing label" in error for error in report.errors)


def test_orphan_label(tmp_path):
    make_dataset(tmp_path)

    label = tmp_path / "train" / "labels" / "orphan.txt"
    label.write_text("1 0.5 0.5 0.2 0.2\n")

    report = validate_dataset(tmp_path)

    assert not report.valid
    assert any("Orphan label" in error for error in report.errors)
