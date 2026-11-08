"""Golden Humble wire image and native ROS serializer parity, without weights."""
from types import SimpleNamespace as Record
import unittest
from hack_perception.cdr_image import decode

# rclpy.serialization.serialize_message(sensor_msgs/Image), ROS Humble/FastDDS.
GOLDEN=bytes.fromhex('00010000070000002a000000080000006f70746963616c000200000002000000050000007267623800000000060000000c000000000102030405060708090a0b')
header=lambda sec,ns,frame:Record(stamp=Record(sec=sec,nanosec=ns),frame_id=frame)


class CDRImageTest(unittest.TestCase):
    def test_humble_golden_wire_image(self):
        image=decode(GOLDEN,header)
        self.assertEqual((image.header.stamp.sec,image.header.stamp.nanosec,image.header.frame_id),(7,42,'optical'))
        self.assertEqual((image.height,image.width,image.encoding,image.step),(2,2,'rgb8',6))
        self.assertEqual(bytes(image.data),bytes(range(12)))
        self.assertIsInstance(image.data,memoryview)

    def test_big_endian_cdr_metadata_has_the_same_image(self):
        wire=bytearray(GOLDEN);wire[:4]=b'\x00\x00\x00\x00'
        for offset in (4,8,12,24,28,32,44,48):wire[offset:offset+4]=wire[offset:offset+4][::-1]
        image=decode(wire,header)
        self.assertEqual((image.height,image.width,image.header.stamp.sec),(2,2,7))
        self.assertEqual(bytes(image.data),bytes(range(12)))

    def test_truncated_and_unknown_encapsulation_are_rejected(self):
        for wire in (GOLDEN[:10],GOLDEN[:-1],b'\x00\x03\x00\x00'+GOLDEN[4:]):
            with self.assertRaises(ValueError):decode(wire,header)

    def test_native_ros_serializer_parity_for_padding_and_pixel_endianness(self):
        try:
            from rclpy.serialization import serialize_message
            from sensor_msgs.msg import Image
            from std_msgs.msg import Header
            from builtin_interfaces.msg import Time
        except ImportError:
            self.skipTest('requires real ROS serializer')
        for frame in ('optical','long_camera_frame','камера'):
            original=Image(header=Header(stamp=Time(sec=9,nanosec=123),frame_id=frame),
                height=2,width=3,encoding='16UC1',is_bigendian=1,step=8,data=list(range(16)))
            image=decode(serialize_message(original),header)
            self.assertEqual((image.header.frame_id,image.step,image.encoding,image.is_bigendian),
                             (frame,8,'16UC1',1))
            self.assertEqual(bytes(image.data),bytes(original.data))
