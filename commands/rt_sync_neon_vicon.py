import time

from pylsl import StreamInfo, StreamOutlet, local_clock
from vicon_dssdk import ViconDataStream

from pupil_labs.realtime_api.simple import Device

# IP address and port of the computer running Vicon Tracker
vicon_hostName = "localhost:801"

client = ViconDataStream.Client()
client.Connect(vicon_hostName)

if not client.IsConnected():
    raise RuntimeError("Could not connect to Vicon Tracker server")

client.EnableSegmentData()
client.SetBufferSize(1)  # Always return most recent frame
client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)

# device = discover_one_device()
device = Device("192.168.1.10", 8080)

estimate = device.estimate_time_offset()
clock_offset_ns = round(estimate.time_offset_ms.mean * 1_000_000)


class ViconLSL:
    def __init__(self):
        self.srate = 0
        self.name = "Vicon"
        self.channel_type = "Markers"
        self.n_channels = 1
        self.offset = 0

    def set_srate(self, srate):
        self.srate = srate

    def get_lsl_to_unix_offset_ns(self):
        lsl_ns = int(local_clock() * 1e9)
        self.offset = time.time_ns() - lsl_ns

    def make_outlet(self):
        self.info = StreamInfo(
            self.name,
            self.channel_type,
            self.n_channels,
            self.srate,
            "string",
            "ViconEventsUID",
        )

        # next make an outlet
        self.outlet = StreamOutlet(self.info)

    def send_event(self, event_text, timestamp):
        # Send an initial sample right away so LabRecorder/other resolvers can see
        # activity.
        try:
            self.outlet.push_sample([event_text], timestamp)
        except Exception as e:
            print("[PSY] initial push_sample failed:", repr(e))

        # Give the resolver a moment to discover the stream (helps on some systems).
        time.sleep(0.2)


vicon_lsl = ViconLSL()
vicon_lsl.make_outlet()

print("")
print("")
print("")
input(
    "Please choose the Vicon LSL and Neon_Event streams in LabRecorder, \
    then start the LabRecorder recording and return here to press Enter \
    to continue. "
)
print("")
print("")
print("")

try:
    INTERVAL_NS = 30 * 1e9
    ref_time = time.time_ns()
    vicon_latencies = []
    srate_set = False
    vicon_frate = 0
    vicon_latency = 0

    while True:
        client.GetFrame()

        current_lsl_time = local_clock()
        current_time_ns_in_client_clock = time.time_ns()
        if abs(current_time_ns_in_client_clock - ref_time) > INTERVAL_NS:
            vicon_frame_num = client.GetFrameNumber()

            if not srate_set:
                vicon_frate = client.GetFrameRate()
                vicon_latency = client.GetLatencyTotal()

                srate_set = True

            event_name = f"Vicon Sync:Frame:{vicon_frame_num}:Frate:{vicon_frate}"

            vicon_lsl.send_event(event_name, current_lsl_time - vicon_latency)
            device.send_event(
                event_name,
                event_timestamp_unix_ns=current_time_ns_in_client_clock
                - clock_offset_ns
                - vicon_latency * 1e9,
            )

            ref_time = time.time_ns()

            vicon_latencies.append(vicon_latency)

except KeyboardInterrupt:
    pass
finally:
    device.close()
    client.Disconnect()
