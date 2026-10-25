import pandas as pd
import pytest

from src.data_loader import (CREDITS_FILENAME, DATA_DIR_ENV, MOVIES_FILENAME, DatasetFormatError,
                             DatasetNotFoundError, load_raw_data, missing_files, resolve_data_dir)


def test_missing_files_raise_with_filenames(tmp_path):
    with pytest.raises(DatasetNotFoundError) as exc:
        load_raw_data(tmp_path)
    assert set(exc.value.missing) == {MOVIES_FILENAME, CREDITS_FILENAME}
    assert MOVIES_FILENAME in str(exc.value)


def test_one_missing_file_is_reported(csv_dir):
    (csv_dir / CREDITS_FILENAME).unlink()
    assert missing_files(csv_dir) == [CREDITS_FILENAME]
    with pytest.raises(DatasetNotFoundError):
        load_raw_data(csv_dir)


def test_valid_files_load(csv_dir):
    movies, credits = load_raw_data(csv_dir)
    assert len(movies) == 11 and len(credits) == 9
    assert {"id", "title", "genres"} <= set(movies.columns)


def test_missing_required_column_is_format_error(csv_dir):
    pd.read_csv(csv_dir / MOVIES_FILENAME).drop(columns=["genres"]).to_csv(csv_dir / MOVIES_FILENAME, index=False)
    with pytest.raises(DatasetFormatError, match="genres"):
        load_raw_data(csv_dir)


def test_empty_file_is_format_error(csv_dir):
    (csv_dir / MOVIES_FILENAME).write_text("")
    with pytest.raises(DatasetFormatError):
        load_raw_data(csv_dir)


def test_header_only_file_is_format_error(csv_dir):
    header = ",".join(pd.read_csv(csv_dir / MOVIES_FILENAME).columns)
    (csv_dir / MOVIES_FILENAME).write_text(header + "\n")
    with pytest.raises(DatasetFormatError, match="no rows"):
        load_raw_data(csv_dir)


def test_environment_variable_selects_directory(monkeypatch, tmp_path):
    monkeypatch.setenv(DATA_DIR_ENV, str(tmp_path))
    assert resolve_data_dir() == tmp_path
    assert resolve_data_dir("/elsewhere").as_posix().endswith("elsewhere")
