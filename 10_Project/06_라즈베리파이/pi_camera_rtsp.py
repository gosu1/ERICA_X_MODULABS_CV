#!/usr/bin/env python3
"""Logitech C270 -> H.264 -> RTSP, served directly by Raspberry Pi 4B.

Raspberry Pi OS (system Python):
  sudo apt install -y python3-gi gir1.2-gstreamer-1.0 \
    gir1.2-gst-rtsp-server-1.0 gstreamer1.0-tools \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly

  /usr/bin/python3 pi_camera_rtsp.py

Client: rtsp://<RASPBERRY_PI_WIFI_IP>:8554/camera
RTSP is not directly playable in a standard web browser.
No access control or TLS; use only on a trusted private network.
"""

import argparse
from pathlib import Path
import sys

try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstRtspServer", "1.0")
    from gi.repository import GLib, Gst, GstRtspServer
except (ImportError, ValueError) as exc:
    sys.exit(
        "GStreamer RTSP Python bindings are missing. Install the packages "
        "shown at the top of this file using apt. " + str(exc)
    )


def make_pipeline(args):
    # Use MJPEG from Logitech C270 so webcam USB traffic is compressed.
    # JPEG must be decoded once before H.264 encoding.
    src = (
        f"v4l2src device={args.device} do-timestamp=true ! "
        f"image/jpeg,width={args.width},height={args.height},"
        f"framerate={args.fps}/1 ! "
        "jpegdec ! videoconvert ! "
    )
    if args.encoder == "hardware":
        # Pi 4B V4L2 H.264 hardware encoder. Some OS/kernel builds can
        # fail to negotiate caps; use --encoder software if that happens.
        encoder = (
            "video/x-raw,format=NV12 ! "
            f'v4l2h264enc extra-controls="controls,video_bitrate={args.bitrate * 1000},'
            f'repeat_sequence_header=1,h264_i_frame_period={args.fps * 2}" ! '
            "video/x-h264,level=(string)4 ! "
        )
    else:
        # Reliable fallback for OS versions without a working V4L2 encoder.
        encoder = (
            "video/x-raw,format=I420 ! "
            "queue max-size-buffers=2 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
            f"x264enc bitrate={args.bitrate} speed-preset=ultrafast "
            f"tune=zerolatency key-int-max={args.fps * 2} "
            "bframes=0 threads=2 byte-stream=true ! "
        )
    tail = (
        "h264parse config-interval=-1 ! "
        "rtph264pay name=pay0 pt=96 config-interval=1"
    )
    return f"( {src}{encoder}{tail} )"


def main():
    parser = argparse.ArgumentParser(description="USB webcam H.264/RTSP server")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--mount", default="/camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--bitrate", type=int, default=2000, help="H.264 kbit/s, default 2000")
    parser.add_argument("--encoder", choices=("software", "hardware"),
                        default="software", help="software (default) or Pi 4 hardware")
    args = parser.parse_args()

    if not Path(args.device).exists():
        parser.error(f"Camera device not found: {args.device}")
    if (args.width < 16 or args.height < 16 or not 1 <= args.fps <= 60
            or not 100 <= args.bitrate <= 25000 or not 1 <= args.port <= 65535):
        parser.error("Invalid width, height, fps, bitrate, or port")
    if not args.mount.startswith("/") or " " in args.mount:
        parser.error("--mount must start with '/' and contain no spaces")

    Gst.init(None)
    necessary = ["v4l2src", "jpegdec", "videoconvert", "h264parse", "rtph264pay"]
    necessary.append("x264enc" if args.encoder == "software" else "v4l2h264enc")
    missing = [name for name in necessary if Gst.ElementFactory.find(name) is None]
    if missing:
        sys.exit("Missing GStreamer plugins: " + ", ".join(missing) +
                 ". Install GStreamer plugins listed in the file header.")

    pipeline = make_pipeline(args)
    try:
        # Catch invalid element properties before accepting connections.
        parsed = Gst.parse_launch(pipeline)
        parsed.set_state(Gst.State.NULL)
    except GLib.Error as exc:
        sys.exit(f"Cannot construct pipeline: {exc}")

    server = GstRtspServer.RTSPServer()
    server.set_service(str(args.port))
    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_launch(pipeline)
    factory.set_shared(True)
    server.get_mount_points().add_factory(args.mount, factory)
    source_id = server.attach(None)
    if not source_id:
        sys.exit(f"Could not start RTSP server on port {args.port}")

    print("[RTSP server started]", flush=True)
    print(f"Stream : rtsp://<PI_WLAN_IP>:{args.port}{args.mount}", flush=True)
    print(f"Webcam : {args.device} | {args.width}x{args.height} "
          f"@ {args.fps} FPS", flush=True)
    print(f"Encoder: {args.encoder} | target {args.bitrate} kbit/s", flush=True)
    print("The camera starts capturing when an RTSP client connects.", flush=True)
    print("Stop: Ctrl+C | trusted LAN only, no authentication", flush=True)
    if args.encoder == "hardware":
        print("If client playback fails, try --encoder software instead.", flush=True)
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopped", flush=True)


if __name__ == "__main__":
    main()
