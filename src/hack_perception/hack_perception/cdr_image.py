"""Bounded XCDR1 sensor_msgs/Image decoding without a Python integer per pixel.

Header/metadata follow the ROS Image IDL. Payload remains a view of the owned
serialized bytes. CameraInfo and all output messages keep normal ROS decoding.
"""
from dataclasses import dataclass
import struct


@dataclass
class ImageFrame:
    header: object
    height: int
    width: int
    encoding: str
    is_bigendian: int
    step: int
    data: memoryview


def decode(data, header_factory):
    view = memoryview(data)
    if len(view) < 4 or bytes(view[:2]) not in (b'\x00\x01', b'\x00\x00'):
        raise ValueError('expected ROS XCDR1 Image encapsulation')
    endian = '<' if view[1] == 1 else '>'
    offset = 4

    def integer(signed=False):
        nonlocal offset
        offset = (offset+3) & ~3
        if offset+4 > len(view):
            raise ValueError('truncated Image metadata')
        value = struct.unpack_from(endian+('i' if signed else 'I'), view, offset)[0]
        offset += 4
        return value

    def string():
        nonlocal offset
        size = integer()
        if not 1 <= size <= 4096 or offset+size > len(view) or view[offset+size-1] != 0:
            raise ValueError('invalid Image metadata string')
        value = bytes(view[offset:offset+size-1]).decode('utf-8')
        offset += size
        return value

    seconds, nanoseconds = integer(True), integer()
    frame = string()
    height, width, encoding = integer(), integer(), string()
    if offset >= len(view):
        raise ValueError('truncated Image byte order')
    bigendian = int(view[offset])
    offset += 1
    step, size = integer(), integer()
    if (not 0 <= nanoseconds < 10**9 or not frame or not height or not width
            or bigendian not in (0, 1) or not step or size != height*step
            or offset+size > len(view) or len(view)-offset-size > 3):
        raise ValueError('invalid Image dimensions, timestamp or payload length')
    return ImageFrame(header_factory(seconds, nanoseconds, frame), height, width,
                      encoding, bigendian, step, view[offset:offset+size])


def ros_image(data):
    from builtin_interfaces.msg import Time
    from std_msgs.msg import Header
    return decode(data, lambda sec, ns, frame: Header(stamp=Time(sec=sec, nanosec=ns), frame_id=frame))
