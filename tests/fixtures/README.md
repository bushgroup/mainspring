# Test fixtures

Nothing here is an acquisition. Sample `.uimf` files live in the private lab repository
(`data/README.md` there is the inventory), PNNL's small test excerpts are fetched into the
gitignored `external/pnnl-testdata/` by `tools/fetch_testdata.py`, and the tests that need
either report themselves skipped when it is absent.

Synthetic files are written at test time by `tests/synthetic.py`, which puts known points
through the real SQLite schema and the real encoder in `mainspring.uimf.decode`, so a
fresh clone still exercises the decode path end to end with no data at all.
