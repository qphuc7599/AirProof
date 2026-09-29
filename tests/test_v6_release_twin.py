import unittest
import numpy as np
from airproof.v6_release_twin import *

class ReleaseTests(unittest.TestCase):
    def calibration(self):
        rng = np.random.default_rng(8)
        truth = rng.normal(size=(2, 50, 2))
        return fit_release_calibration(truth, np.zeros_like(truth), truth+.1*rng.normal(size=truth.shape))

    def test_var_and_public_equivalence(self):
        c = self.calibration()
        self.assertLessEqual(np.linalg.norm(c.transition, 2), .98000001)
        a, b = ReleaseOnlyTwin(c), PublicCalibratedTwin(c)
        for e in range(25):
            np.testing.assert_array_equal(a.update(e, np.ones(2)), b.update(e, np.ones(2)))

    def test_duplicate_and_frozen_live(self):
        a = ReleaseOnlyTwin(self.calibration())
        for e in range(24): a.update(e, np.ones(2))
        frozen = a.live[0].copy()
        r = PublicReleaseObservation('a', 0, 24, 0, 2., 7.)
        a.update(24, np.ones(2), [r, r])
        self.assertEqual(a.assimilated, 1)
        self.assertEqual(a.duplicates, 1)
        np.testing.assert_array_equal(a.live[0], frozen)
        self.assertTrue(a.diagnostics()['dp_variance_added_once'])

    def test_covariance_and_clock(self):
        from dataclasses import replace
        c = self.calibration()
        bad = c.initial_covariance.copy(); bad[0, 1] += 1
        with self.assertRaises(ValueError): ReleaseOnlyTwin(replace(c, initial_covariance=bad))
        a = ReleaseOnlyTwin(c)
        with self.assertRaises(ValueError):
            a.update(0, np.ones(2), [PublicReleaseObservation('a', 0, 0, 0, 1., 7.)])
        self.assertEqual(a.epochs, [])

    def test_noise_added_exactly_once(self):
        from airproof.v5_release_twin import ReleaseCalibration
        c = ReleaseCalibration(np.zeros(2), np.eye(2)*.5, np.eye(2), np.eye(2), 1)
        a = ReleaseOnlyTwin(c)
        for e in range(24): a.update(e, np.zeros(1))
        a.update(24, np.zeros(1), [PublicReleaseObservation('a', 0, 24, 0, 0., 7.)])
        self.assertAlmostEqual(a.covariance[0, 0], 1-1/(2+98))

    def test_suppression_is_missing_and_release_slot_has_one_effect(self):
        c=self.calibration(); missing=ReleaseOnlyTwin(c); suppressed=ReleaseOnlyTwin(c)
        for e in range(24):
            missing.update(e,np.ones(2));suppressed.update(e,np.ones(2))
        expected=missing.update(24,np.ones(2))
        actual=suppressed.update(24,np.ones(2),[
            PublicReleaseObservation('slot-0',0,24,0,None,7.)])
        np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(suppressed.covariance,missing.covariance)
        self.assertEqual(suppressed.suppressed,1)
        conflicting=ReleaseOnlyTwin(c)
        for e in range(24):conflicting.update(e,np.ones(2))
        with self.assertRaises(ValueError):
            conflicting.update(24,np.ones(2),[
                PublicReleaseObservation('first-id',0,24,0,3.,7.),
                PublicReleaseObservation('different-id',0,24,0,4.,7.)])
        self.assertEqual(conflicting.epochs,list(range(24)))

    def test_duplicate_identifier_in_one_publication_assimilates_once(self):
        a=ReleaseOnlyTwin(self.calibration())
        for e in range(24):a.update(e,np.ones(2))
        r=PublicReleaseObservation('one-effect',0,24,0,2.,7.)
        a.update(24,np.ones(2),[r,r])
        self.assertEqual(a.assimilated,1)
        self.assertEqual(a.duplicates,1)
        with self.assertRaises(ValueError):
            ReleaseOnlyTwin(self.calibration(),lag=47)

    def test_diagnostic_support_guard(self):
        with self.assertRaises(ValueError): information_diagnostics([1], [1,2], [1], [1])

class SparseAndLaplaceTests(unittest.TestCase):
    def test_laplace_symmetric_and_limits(self):
        center = scalar_laplace_reference(3., 4., 3., 2.)
        self.assertAlmostEqual(center['mean'], 3.)
        self.assertGreater(center['variance'], 0.)
        self.assertLess(center['variance'], 4.)
        tiny = scalar_laplace_reference(0., 1., 2., 1e-4)
        self.assertAlmostEqual(tiny['mean'], 2., places=6)
        self.assertAlmostEqual(tiny['variance']/2e-8, 1., places=5)
        large = scalar_laplace_reference(0., 1., 2., 1e6)
        self.assertAlmostEqual(large['mean'], 0., places=5)
        self.assertAlmostEqual(large['variance'], 1., places=5)
        self.assertLess(center['normalization_relative_error_estimate'], 1e-8)
        # For centered unit prior, known normal-Laplace density at zero.
        from scipy.special import erfc
        r = scalar_laplace_reference(0.,1.,0.,1.)
        self.assertAlmostEqual(np.exp(r['log_normalization']), .5*np.exp(.5)*erfc(1/np.sqrt(2)), places=10)

    def test_sparse_actual_slots(self):
        rng = np.random.default_rng(50)
        t = rng.normal(size=(3,85,2)); p = np.zeros_like(t)
        mask = np.zeros(t.shape[:2],bool); mask[:,::24] = True
        q = np.full_like(t,np.nan)
        q[mask] = t[mask]+.4*t[mask]+rng.normal(scale=.1,size=t[mask].shape)
        c = fit_sparse_release_calibration(t,p,q,mask)
        self.assertEqual(c.scheduled_samples,12)
        self.assertTrue(np.any(abs(c.initial_covariance[:2,2:]) > .01))
        self.assertTrue(np.all(c.transition[2:] == 0))
        ReleaseOnlyTwin(c)
        q[~mask] = 0
        with self.assertRaises(ValueError): fit_sparse_release_calibration(t,p,q,mask)
