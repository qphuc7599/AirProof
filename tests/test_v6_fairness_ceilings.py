import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_v6_fairness_ceilings import perfect_distinct_allocation
from scripts.diagnose_v6_fairness_ceilings import gap


ROOT = Path(__file__).resolve().parents[1]


def test_gap_is_raw_count_composition_sensitive():
    assert gap({0: 80, 1: 100}) == pytest.approx(.2)
    assert gap({0: 100, 1: 100}) == 0


def test_perfect_distinct_allocation_matches_exhaustive_small_fixture():
    # Exact total four: (1, 1, 2) is optimal under caps (1, 2, 3).
    assert perfect_distinct_allocation([1, 2, 3], 4) == {
        'gap': .5, 'minimum': 1, 'maximum': 2}


def test_exposed_certificate_retains_both_failed_ceilings_and_hashes():
    path = ROOT/'reports/v6/fairness_ceiling_diagnostic/certificate.json'
    certificate = json.loads(path.read_text())
    assert certificate['H6']['all_population_contrasts_positive']
    assert certificate['H6']['all_timely_contrasts_positive']
    assert certificate['H6']['all_fixed_capacity_oracle_contrasts_positive']
    assert certificate['H7']['oracle_selected_alpha_zero_worlds'] == 12
    assert certificate['H7']['oracle_contrast']['minimum'] > 0
    assert certificate['decision'].startswith('do_not_open')
    outcomes = ROOT/'reports/v6/fairness_ceiling_diagnostic/outcomes.json'
    assert hashlib.sha256(outcomes.read_bytes()).hexdigest() == certificate['source_sha256']['ceiling_outcomes']
