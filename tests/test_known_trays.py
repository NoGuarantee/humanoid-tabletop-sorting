from types import SimpleNamespace as NS
import unittest

from hack_brain.sensors import with_known_trays


def tray(tray_id, x):
    return NS(id=tray_id, kind='tray', x=x)


class KnownTrays(unittest.TestCase):
    def test_hidden_tray_comes_from_the_last_look(self):
        known = {}
        with_known_trays(NS(containers=[tray('tray_left', 1), tray('tray_center', 2)]), known)
        scene = with_known_trays(NS(containers=[tray('tray_center', 3)]), known)
        self.assertEqual(sorted((x.id, x.x) for x in scene.containers), [('tray_center', 3), ('tray_left', 1)])

    def test_a_tray_in_view_replaces_the_remembered_one(self):
        known = {}
        with_known_trays(NS(containers=[tray('tray_left', 1)]), known)
        with_known_trays(NS(containers=[tray('tray_left', 4)]), known)
        scene = with_known_trays(NS(containers=[]), known)
        self.assertEqual([(x.id, x.x) for x in scene.containers], [('tray_left', 4)])

    def test_racks_are_not_remembered(self):
        known = {}
        with_known_trays(NS(containers=[NS(id='rack_a', kind='rack')]), known)
        self.assertEqual(with_known_trays(NS(containers=[]), known).containers, [])


if __name__ == '__main__':
    unittest.main()
