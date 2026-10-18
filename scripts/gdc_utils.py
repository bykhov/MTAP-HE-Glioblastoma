"""
Small helpers for the public GDC API (https://api.gdc.cancer.gov). Only open-access files are used, so no token
or credentials are involved. Used by ``cohort_copy_number.py``.
"""
import json
import time

import requests

GDC_API = "https://api.gdc.cancer.gov"
FILE_FIELDS = "file_id,file_name,file_size,md5sum,access,data_type,data_format,analysis.workflow_type,cases.submitter_id"


def query_files(filters, fields=FILE_FIELDS, page_size=1000, api=GDC_API, session=requests):
    """All file records matching ``filters`` (a GDC filter dict), following the pagination."""
    hits, start = [], 0
    while True:
        reply = session.get(f"{api}/files", params={"filters": json.dumps(filters), "fields": fields,
                                                    "size": page_size, "from": start, "format": "json"}, timeout=120)
        reply.raise_for_status()
        data = reply.json()["data"]
        hits += data["hits"]
        start += page_size
        if start >= data["pagination"]["total"] or not data["hits"]:
            return hits


def in_filter(field, values):
    return {"op": "in", "content": {"field": field, "value": sorted(values)}}


def equals_filter(field, value):
    return {"op": "=", "content": {"field": field, "value": value}}


def and_filter(*parts):
    return {"op": "and", "content": list(parts)}


def with_retries(func, attempts=6, wait=2.0, log=print):
    """Call ``func()``; on a network error wait (2, 4, 8, ... s, at most 60) and try again, then raise the last error."""
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except (requests.RequestException, OSError) as error:
            if attempt == attempts:
                raise
            log(f"  network error ({error.__class__.__name__}), retry {attempt}/{attempts - 1}")
            time.sleep(min(wait * 2 ** (attempt - 1), 60))


def stream_tsv_row(file_id, key_column, key_value, api=GDC_API, session=requests):
    """
    Read a GDC TSV as a stream and return ``{column: value}`` for the first row whose ``key_column`` equals
    ``key_value`` (the connection is closed as soon as the row is found). ``None`` if the row does not exist.
    """
    with session.get(f"{api}/data/{file_id}", stream=True, timeout=120) as reply:
        reply.raise_for_status()
        reply.encoding = "utf-8"  # the GDC sends no charset; without it iter_lines yields bytes
        header = None
        for raw in reply.iter_lines(decode_unicode=True):
            if not raw:
                continue
            fields = raw.split("\t")
            if header is None:
                header = fields
                key_index = header.index(key_column)
                continue
            if fields[key_index] == key_value:
                return dict(zip(header, fields))
    return None
