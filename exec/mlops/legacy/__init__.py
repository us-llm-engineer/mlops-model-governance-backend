"""Project 1's R1-R3 deliverable (api/auth/orchestration/serving/metrics), kept as a
closed, graded regression baseline -- not live product code. Its 236 tests are kept as
a regression suite.

Isolated here (connect-pipeline sweep) so the package layout makes clear which code
is the live service (exec/mlops/svc/, interop/, ext/, and the core domain modules
imported by svc/app.py) and which is preserved prior-round work with no production
caller. mlops/__init__.py re-exports every public name from here unchanged, so
`import mlops; mlops.ArtifactStore` etc. still work exactly as before this move.
"""
