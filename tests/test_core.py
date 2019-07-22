from unittest import TestCase

from sister import Core


class CoreTextCase(TestCase):

    def setup(self):
        pass

    def tearDown(self):
        pass

    def test_main(self):
        name = 'sotaro'
        core = Core()
        self.assertEqual(f'Hello {name}!!!', core.main(name))
