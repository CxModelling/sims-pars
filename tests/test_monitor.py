import logging
import unittest

from sims_pars.monitor import Monitor

__author__ = 'TimeWz667'


class TestMonitor(unittest.TestCase):
    def setUp(self):
        self.Mon = Monitor('Test')

    def test_recording(self):
        self.Mon.keep(a=1)
        self.assertEqual(self.Mon['a'], 1)
        self.Mon.step()
        self.assertEqual(len(self.Mon.Records), 1)
        self.assertEqual(self.Mon.Time, 1)

        with self.assertRaises(KeyError):
            self.Mon.step(0.5)

    def test_reset(self):
        self.Mon.reset(50)
        self.assertEqual(self.Mon.Time, 50)
        self.Mon.keep(a=1)
        self.assertEqual(self.Mon['a'], 1)
        self.Mon.step(53)
        self.assertEqual(len(self.Mon.Records), 1)
        self.assertEqual(self.Mon.Time, 53)


def test_trajectories_shape():
    mon = Monitor('traj-test', stream_handler=False)
    mon.keep(Size=4)
    mon.step()
    mon.keep(Size=6)
    mon.step()
    df = mon.Trajectories
    assert list(df['Size']) == [4, 6]
    assert df.index.name == 'Time'


def test_no_duplicate_stream_handlers():
    name = 'dedup-test'
    Monitor(name)
    Monitor(name)
    Monitor(name)
    logger = logging.getLogger(name)
    streams = [h for h in logger.handlers
               if isinstance(h, logging.StreamHandler)
               and not isinstance(h, logging.FileHandler)]
    assert len(streams) == 1


def test_set_log_path_is_idempotent(tmp_path):
    name = 'file-dedup-test'
    mon = Monitor(name, stream_handler=False)
    path = str(tmp_path / 'run.log')
    mon.set_log_path(path)
    mon.set_log_path(path)
    files = [h for h in logging.getLogger(name).handlers
             if isinstance(h, logging.FileHandler)]
    assert len(files) == 1
