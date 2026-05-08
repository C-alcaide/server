using System;
using System.Runtime.InteropServices;

class Program {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct DISPLAY_DEVICE {
        public int cb;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
        public string DeviceName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string DeviceString;
        public uint StateFlags;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string DeviceID;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
        public string DeviceKey;
    }

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    static extern bool EnumDisplayDevices(string lpDevice, uint iDevNum, ref DISPLAY_DEVICE lpDisplayDevice, uint dwFlags);

    static void Main() {
        DISPLAY_DEVICE dd = new DISPLAY_DEVICE();
        dd.cb = Marshal.SizeOf(dd);
        
        for (uint i = 0; EnumDisplayDevices(null, i, ref dd, 0); i++) {
            string attached = (dd.StateFlags & 1) != 0 ? "ATTACHED" : "detached";
            Console.WriteLine(string.Format("Adapter {0}: {1,-15} | {2,-30} | {3}", i, dd.DeviceName, dd.DeviceString, attached));
            
            if ((dd.StateFlags & 1) != 0) {
                DISPLAY_DEVICE mon = new DISPLAY_DEVICE();
                mon.cb = Marshal.SizeOf(mon);
                if (EnumDisplayDevices(dd.DeviceName, 0, ref mon, 0)) {
                    Console.WriteLine(string.Format("  Monitor: {0} | ID: {1}", mon.DeviceString, mon.DeviceID));
                }
            }
            dd.cb = Marshal.SizeOf(dd);
        }
    }
}
