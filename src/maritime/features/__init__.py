"""Behavioral features computed from canonical AIS (milestone 2).

Planned, all region-agnostic and later scored against a local (cell x vessel type)
baseline:
- AIS gaps (and expected coverage per cell, so sparse receivers are not flagged)
- loitering
- vessel-to-vessel encounters (candidate ship-to-ship transfers)
- draught changes without a port call
- kinematic impossibilities (spoofing tells)
- port visits

Definitions and thresholds belong in docs/design.md before code goes here.
"""
