"""Where the server keeps uploads, run state and reports: one data directory, owner-only.

    <data_dir>/uploads/<id>/records.jsonl       the validated records (not the archive itself)
    <data_dir>/uploads/<id>/info.json           an `UploadInfo`
    <data_dir>/runs/<id>/state.json             a `RunStatus`
    <data_dir>/runs/<id>/report.json            the finished `Report`

Ids are random (`uuid4().hex`) and are the only thing ever used to build a path: `_checked_id`
rejects anything else, so a request cannot reach outside the data directory. The directory and
every file in it are readable by the owner only (0700 and 0600), because they hold partner data.

Open decisions (see the notes on RUN-7):
    * how long uploads are kept. The MVP deletes only when the user asks; RUN-7's rule for the
      hosted service is to delete the worker's data once the report is written.
    * a run that was `running` when the server stopped has no process behind it any more; on
      start-up `Store.list_runs` callers should mark such runs `failed` ("the server restarted")
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from migration_eval.archive import component_summaries
from migration_eval.capture.sink import record_to_json
from migration_eval.models import CallRecord, Report
from migration_eval.run import write_report
from migration_eval.server.models import ComponentInfo, RunStatus, UploadInfo

_ID = re.compile(r"[0-9a-f]{32}")
MAX_FILENAME_SHOWN = 200


class NotFound(LookupError):
    """No upload or run with that id."""


def _checked_id(value: str) -> str:
    """The id if it is one of ours, else NotFound: ids are the only thing used to build a path."""
    if not _ID.fullmatch(value):
        raise NotFound(value)
    return value


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)  # mkdir's mode is cut down by the umask and ignored for an existing folder
    return path


def _write_private(path: Path, data: bytes) -> None:
    """Write `data` to `path` (owner-only) through a temporary file, so a reader never sees half of it."""
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")  # created 0600
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _display_name(filename: str) -> str:
    """The upload's name for the screen only: no folders, no control characters, not too long."""
    name = Path(filename.replace("\\", "/")).name
    return "".join(ch for ch in name if ch.isprintable())[:MAX_FILENAME_SHOWN] or "upload"


class Store:
    def __init__(self, data_dir: Path) -> None:
        """Create `data_dir` with its `uploads` and `runs` folders, mode 0700, if they are missing."""
        self.root = _private_dir(Path(data_dir))
        self._uploads = _private_dir(self.root / "uploads")
        self._runs = _private_dir(self.root / "runs")

    def _folder(self, base: Path, item_id: str) -> Path:
        folder = base / _checked_id(item_id)
        if not folder.is_dir():
            raise NotFound(item_id)
        return folder

    # uploads
    def save_upload(self, filename: str, size_bytes: int, records: list[CallRecord]) -> UploadInfo:
        """Give the upload a new id, write its records and its `UploadInfo` (via
        `archive.component_summaries`), and return the info."""
        upload_id = uuid.uuid4().hex
        folder = _private_dir(self._uploads / upload_id)
        started = [r.started_at for r in records]
        info = UploadInfo(
            id=upload_id,
            filename=_display_name(filename),
            size_bytes=size_bytes,
            record_count=len(records),
            first_started_at=min(started, default=None),
            last_started_at=max(started, default=None),
            components=[
                ComponentInfo(
                    name=c.name,
                    records=c.records,
                    replay_grade=c.replay_grade,
                    incumbent_provider=c.incumbent.provider if c.incumbent else None,
                    incumbent_model=c.incumbent.model if c.incumbent else None,
                )
                for c in component_summaries(records)
            ],
        )
        _write_private(folder / "records.jsonl", "".join(record_to_json(r) + "\n" for r in records).encode("utf-8"))
        _write_private(folder / "info.json", info.model_dump_json().encode("utf-8"))  # last: its presence means "complete"
        return info

    def get_upload(self, upload_id: str) -> UploadInfo:
        """The stored info. Raise `NotFound`."""
        folder = self._folder(self._uploads, upload_id)
        try:
            return UploadInfo.model_validate_json((folder / "info.json").read_bytes())
        except OSError:
            raise NotFound(upload_id) from None

    def load_records(self, upload_id: str) -> list[CallRecord]:
        """The upload's records, for starting a run. Raise `NotFound`."""
        folder = self._folder(self._uploads, upload_id)
        try:
            lines = (folder / "records.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError:
            raise NotFound(upload_id) from None
        return [CallRecord.model_validate_json(line) for line in lines if line.strip()]

    def delete_upload(self, upload_id: str) -> None:
        """Remove the upload's folder. Raise `NotFound`."""
        shutil.rmtree(self._folder(self._uploads, upload_id))

    # runs
    def save_run(self, status: RunStatus) -> None:
        """Write (replace) `runs/<id>/state.json`. Called on every stage change, so write to a
        temporary file and rename it: a reader never sees half a file."""
        folder = _private_dir(self._runs / _checked_id(status.id))
        _write_private(folder / "state.json", status.model_dump_json().encode("utf-8"))

    def get_run(self, run_id: str) -> RunStatus:
        """Raise `NotFound`."""
        folder = self._folder(self._runs, run_id)
        try:
            return RunStatus.model_validate_json((folder / "state.json").read_bytes())
        except OSError:
            raise NotFound(run_id) from None

    def list_runs(self) -> list[RunStatus]:
        """Every run, newest first."""
        runs: list[RunStatus] = []
        for folder in self._runs.iterdir():
            try:
                runs.append(self.get_run(folder.name))
            except (NotFound, ValueError):  # a folder with no or damaged state is skipped, not fatal
                continue
        return sorted(runs, key=lambda r: r.started_at, reverse=True)

    def save_report(self, run_id: str, report: Report) -> None:
        """Write the report with `run.write_report` (owner-only)."""
        write_report(report, self._folder(self._runs, run_id) / "report.json")

    def get_report(self, run_id: str) -> Report:
        """Raise `NotFound` until the run has one."""
        folder = self._folder(self._runs, run_id)
        try:
            return Report.model_validate_json((folder / "report.json").read_bytes())
        except OSError:
            raise NotFound(run_id) from None

    def delete_run(self, run_id: str) -> None:
        """Remove the run's folder, its report included. Raise `NotFound`."""
        shutil.rmtree(self._folder(self._runs, run_id))
