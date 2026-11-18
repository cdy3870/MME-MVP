"""The local server behind the UI: upload an export, run an evaluation, read the report.

    migration-eval export      (terminal)  -> an archive
    POST /api/uploads                      -> the server checks it and lists its components
    POST /api/runs                         -> the evaluation runs in the background
    GET  /api/runs/{id}                    -> live stage, spend and state
    GET  /api/runs/{id}/report             -> the report the viewer draws

These are the ERD's `POST /runs` and `GET /runs/{id}` (RUN-1), upload (EXP-4) and manifest
validation (EXP-5), built local-first so the same code can later become the hosted service.

Security rules that hold for every module here:
    * it listens on 127.0.0.1 only, with no login, so it is for the machine you are sitting at;
      hosting it needs accounts and tokens first (ACC-1)
    * provider keys come from the server's environment and never cross the API (`keys.py`)
    * a run starts only when the request says the user confirmed what will be sent and where
    * uploaded data and reports live in owner-only files under one data directory, and can be deleted
    * ids are random and are the only thing used to build a path
"""
