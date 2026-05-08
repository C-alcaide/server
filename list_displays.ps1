Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DispEnum {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct DISPLAY_DEVICE {
        public int cb;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string DeviceName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceString;
        public int StateFlags;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceID;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceKey;
    }
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern bool EnumDisplayDevices(string lpDevice, uint iDevNum, ref DISPLAY_DEVICE lpDisplayDevice, uint dwFlags);
}
"@

for ($i = 0; $i -lt 10; $i++) {
    $dd = New-Object DispEnum+DISPLAY_DEVICE
    $dd.cb = [Runtime.InteropServices.Marshal]::SizeOf($dd)
    if ([DispEnum]::EnumDisplayDevices($null, $i, [ref]$dd, 0)) {
        $mon = New-Object DispEnum+DISPLAY_DEVICE
        $mon.cb = [Runtime.InteropServices.Marshal]::SizeOf($mon)
        $monName = "(none)"
        if ([DispEnum]::EnumDisplayDevices($dd.DeviceName, 0, [ref]$mon, 0)) {
            $monName = $mon.DeviceString
        }
        $att = if ($dd.StateFlags -band 1) { "ATTACHED" } else { "detached" }
        Write-Output ("{0}: {1} | Adapter={2} | Monitor={3} | {4}" -f $i, $dd.DeviceName, $dd.DeviceString, $monName, $att)
    }
}
