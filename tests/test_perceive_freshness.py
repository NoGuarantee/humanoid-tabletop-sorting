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
    from std_msgs.msg import Header
    from builtin_interfaces.msg import Time
    return Record(header=Header(frame_id='head_optical',stamp=Time(sec=stamp//10**9,nanosec=stamp%10**9)))


@unittest.skipIf(rclpy is None, 'requires real ROS Humble imports')
class PerceiveFreshnessTest(unittest.TestCase):
    def setUp(self):
        rclpy.init(args=[])
        class InputContractNode(PerceptionNode):
            def detect(self,f,region,scene,geometry,epoch,timing):
                scene.table_z=.1  # Only test source selection, not geometry.
        self.node=InputContractNode()
        self.node.set_parameters([rclpy.parameter.Parameter('perceive_frame_wait_s',value=.2)])
        self.node.names_ready.set()  # names are tested on their own below

    def tearDown(self):
        self.node.diagnostics.close()
        self.node.destroy_node()
        rclpy.shutdown()

    def test_first_response_waits_for_post_request_complete_exposure(self):
        node=self.node
        past=node.get_clock().now().nanoseconds-100_000_000
        for kind in ('color','depth','color_info','depth_info'):node.store('head',kind,message(past))
        emitted=[]
        def publish():
            time.sleep(.02)
            fresh=node.get_clock().now().nanoseconds
            for kind in ('color','depth'):node.store('head',kind,message(fresh))
            time.sleep(.02)
            emitted.append(fresh)
            node.store('head','color_info',message(fresh))
            node.store('head','depth_info',message(fresh))
        thread=threading.Thread(target=publish);thread.start()
        requested=node.get_clock().now().nanoseconds
        result=node.on_perceive(Perceive.Request(camera='head',region='table'),Perceive.Response())
        thread.join()
        self.assertTrue(result.success,result.message)
        stamp=result.scene.header.stamp.sec*10**9+result.scene.header.stamp.nanosec
        self.assertEqual(stamp,emitted[0])
        self.assertGreaterEqual(stamp,requested)

    def test_without_a_fresh_exposure_the_last_complete_one_is_returned_as_stale(self):
        node=self.node
        old=node.get_clock().now().nanoseconds-3_000_000_000  # older than any source TTL
        for kind in ('color','depth','color_info','depth_info'):node.store('head',kind,message(old))
        node.store('head','color',message(old+1))  # incomplete newer exposure is skipped
        started=time.monotonic()
        result=node.on_perceive(Perceive.Request(camera='head',region='table'),Perceive.Response())
        self.assertTrue(result.success,result.message)
        self.assertTrue(result.message.startswith('stale frame'),result.message)
        stamp=result.scene.header.stamp.sec*10**9+result.scene.header.stamp.nanosec
        self.assertEqual(stamp,old)
        self.assertLess(time.monotonic()-started,.5)

    def test_no_complete_exposure_is_an_error(self):
        node=self.node
        for kind in ('color','depth','color_info'):node.store('head',kind,message(node.get_clock().now().nanoseconds))
        result=node.on_perceive(Perceive.Request(camera='head',region='table'),Perceive.Response())
        self.assertFalse(result.success)
        self.assertIn('no frames',result.message)

    def test_perceive_waits_for_object_names(self):
        from std_msgs.msg import String
        node=self.node
        node.names_ready.clear()
        stamp=node.get_clock().now().nanoseconds-1_000_000
        for kind in ('color','depth','color_info','depth_info'):node.store('head',kind,message(stamp))
        def publish():
            time.sleep(.1)
            node.on_names(String(data='{"Ecoforms_Cup_B4_SAN": "dish"}'))
        thread=threading.Thread(target=publish);thread.start()
        started=time.monotonic()
        result=node.on_perceive(Perceive.Request(camera='head',region='table'),Perceive.Response())
        thread.join()
        self.assertTrue(result.success,result.message)
        self.assertGreaterEqual(time.monotonic()-started,.1)
        self.assertEqual(node.names,{'Ecoforms_Cup_B4_SAN':'dish'})

    def test_mixed_camera_headers_are_not_a_snapshot(self):
        node=self.node
        stamp=node.get_clock().now().nanoseconds
        for kind in ('color','depth','color_info','depth_info'):node.store('head',kind,message(stamp))
        node.store('head','color',message(stamp+1))
        frames,key=node.snapshot('head')
        self.assertEqual(key[1],stamp)
        self.assertEqual(frames['color'].header.stamp,frames['depth'].header.stamp)
