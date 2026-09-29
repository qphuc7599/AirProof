from airproof.validation import run_p0_fixtures


def test_all_p0_gates_pass():
    report = run_p0_fixtures()
    assert report["all"], report
