"""ROS contract tests; synthetic input only, separate from live acceptance."""
import threading
import time
from types import SimpleNamespace as Record
import unittest
try:
    import rclpy
    from hack_interfaces.srv import Perceive
    from hack_perception.perception_node import PerceptionNode
except ImportError:
    rclpy = None


def message(stamp):
    return Record(header=Record(frame_id='head_optical',stamp=Record(sec=stamp//10**9,nanosec=stamp%10**9)))


@unittest.skipIf(rclpy is None, 'requires real ROS Humble imports')
class PerceiveFreshnessTest(unittest.TestCase):
    def setUp(self):
        rclpy.init(args=[])
        class InputContractNode(PerceptionNode):
            def detect(self,f,region,scene,geometry,epoch,timing):
                scene.table_z=.1  # Only test source selection, not geometry.
        self.node=InputContractNode()
        self.node.set_parameters([rclpy.parameter.Parameter('perceive_frame_wait_s',value=.2)])

    def tearDown(self):
        self.node.diagnostics.close()
        self.node.destroy_node()
        rclpy.shutdown()

    def test_first_response_waits_for_post_request_complete_exposure(self):
        node=self.node
        past=node.get_clock().now().nanoseconds-100_000_000
        for kind in ('color','depth','color_info'):node.store('head',kind,message(past))
        emitted=[]
        def publish():
            time.sleep(.02)
            fresh=node.get_clock().now().nanoseconds
            for kind in ('color','depth'):node.store('head',kind,message(fresh))
            time.sleep(.02)
            emitted.append(fresh)
            node.store('head','color_info',message(fresh))
        thread=threading.Thread(target=publish);thread.start()
        requested=node.get_clock().now().nanoseconds
        result=node.on_perceive(Perceive.Request(camera='head',region='table'),Perceive.Response())
        thread.join()
        self.assertTrue(result.success,result.message)
        stamp=result.scene.header.stamp.sec*10**9+result.scene.header.stamp.nanosec
        self.assertEqual(stamp,emitted[0])
        self.assertGreaterEqual(stamp,requested)

    def test_pre_request_exposure_is_not_returned_even_if_under_old_ttl(self):
        node=self.node
        old=node.get_clock().now().nanoseconds-1_000_000
        for kind in ('color','depth','color_info'):node.store('head',kind,message(old))
        started=time.monotonic()
        result=node.on_perceive(Perceive.Request(camera='head',region='table'),Perceive.Response())
        self.assertFalse(result.success)
        self.assertLess(time.monotonic()-started,.5)

    def test_mixed_camera_headers_are_not_a_snapshot(self):
        node=self.node
        stamp=node.get_clock().now().nanoseconds
        for kind in ('color','depth','color_info'):node.store('head',kind,message(stamp))
        node.store('head','color',message(stamp+1))
        self.assertEqual(node.snapshot('head'),(None,None))
