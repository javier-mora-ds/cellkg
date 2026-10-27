"""The replay copies steps from run_pykeen.main(); if the original changes, the replay must be re-audited."""
from gnn.common import sha256_file
from gnn.pykeen_replay import RUN_PYKEEN

RECORDED_SHA256 = "700a72c089ec2831f9f12441d345de90d37ab870b2d8a81ef6e6f12ecd741817"


def test_run_pykeen_unchanged():
    assert sha256_file(RUN_PYKEEN) == RECORDED_SHA256
