import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

print("Devices found:", len(devices))

for dev in devices:
    print(dev.get_info(rs.camera_info.name))