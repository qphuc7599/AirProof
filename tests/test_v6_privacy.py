import unittest
from airproof.privacy import stabilized_bounded_mean
from airproof.v6_privacy import candidate_sensitivity, release_contract_certificate, sensitivity_audit

class SensitivityTests(unittest.TestCase):
    def test_exhaustive_vertices_at_boundaries(self):
        # Permutation invariance means number of positive vertices enumerates all.
        for k in (1,2,3,20):
            C = 10.
            f = lambda xs: stabilized_bounded_mean(xs, clip=C, k_min=k)
            add = 2*C/(k+1)
            for n in set((max(0,k-2), k-1,k,k+1,k+2)):
                for positives in range(n+1):
                    xs = [C]*positives+[-C]*(n-positives)
                    for v in (-C,C):
                        self.assertLessEqual(abs(f(xs+[v])-f(xs)), add+1e-12)
                        for w in (-C,C):
                            self.assertLessEqual(abs(f(xs+[v])-f(xs+[w])), 2*C/k+1e-12)
            # Sharp group switch: remove opposite-sign outlier from each group.
            xs = [-C]*k
            delta = abs(f(xs+[C])-f(xs))*2
            self.assertAlmostEqual(delta, 4*C/(k+1))
            self.assertLessEqual(delta, candidate_sensitivity(clip=C,k_min=k)+1e-12)

    def test_mechanism_unchanged(self):
        audit = sensitivity_audit()
        self.assertAlmostEqual(audit['candidate_scale'], 20/3)
        self.assertEqual(audit['default_v5_scale'], 7)
        self.assertFalse(audit['mechanism_changed'])
        self.assertTrue(audit['actual_admission_composition_verified'])

    def test_deployed_whole_history_certificate_is_exact_and_conservative(self):
        certificate=release_contract_certificate()
        self.assertEqual(certificate['publications'],28)
        self.assertEqual(certificate['k_min'],20)
        self.assertEqual(certificate['deadline_epochs'],24)
        self.assertEqual(certificate['consumer_fixed_lag_epochs'],48)
        self.assertAlmostEqual(certificate['whole_history_epsilon'],8.)
        self.assertAlmostEqual(certificate['deployed_vector_l1_sensitivity'],2.)
        self.assertAlmostEqual(certificate['deployed_laplace_scale'],7.)
        self.assertGreater(certificate['deployed_vector_l1_sensitivity'],
                           certificate['candidate_tighter_bound_not_deployed'])
        with self.assertRaises(ValueError):release_contract_certificate(publications=27)
        with self.assertRaises(ValueError):release_contract_certificate(epsilon_history=8.01)
