import unittest
import numpy as np
from stonesoup.types.orbitalelements import OrbitalElements

class MyTestCase(unittest.TestCase):
    #def test_something(self):
    #    self.assertEqual(True, False)

    def test_orbital_elements(self):
        a = OrbitalElements(np.array([[0.42607], [10.64e6], [39.687/180 *np.pi], [130.32/180* np.pi], [42.373/180* np.pi],
                                     [52.404/180 *np.pi]]))
        r = a.position_vector()
        self.assertAlmostEqual(r[0], -3.67e6, delta=2000)
        self.assertAlmostEqual(r[1], -3.870e6, delta=2000)
        self.assertAlmostEqual(r[2], 4.4e6, delta=2000)

        v = a.velocity_vector()
        self.assertAlmostEqual(v[0], 4.7e3, delta=2)
        self.assertAlmostEqual(v[1], -7.4e3, delta=2)
        self.assertAlmostEqual(v[2], 1e3, delta=2)

if __name__ == '__main__':
    unittest.main()
