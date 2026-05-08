Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DH3 {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
    public struct DEVMODE {
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
        public short dmSpecVersion; public short dmDriverVersion; public short dmSize;
        public short dmDriverExtra; public int dmFields;
        public int dmPositionX; public int dmPositionY;
        public int dmDisplayOrientation; public int dmDisplayFixedOutput;
        public short dmColor; public short dmDuplex; public short dmYResolution;
        public short dmTTOption; public short dmCollate;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
        public short dmLogPixels; public int dmBitsPerPel;
        public int dmPelsWidth; public int dmPelsHeight;
        public int dmDisplayFlags; public int dmDisplayFrequency;
        public int dmICMMethod; public int dmICMIntent; public int dmMediaType;
        public int dmDitherType; public int dmReserved1; public int dmReserved2;
        public int dmPanningWidth; public int dmPanningHeight;
    }
    [DllImport("user32.dll")] public static extern bool EnumDisplaySettingsA(string dev, int mode, ref DEVMODE dm);
    [DllImport("user32.dll")] public static extern int ChangeDisplaySettingsExA(string dev, ref DEVMODE dm, IntPtr hw, int fl, IntPtr lp);
}
"@

# Get current settings
$dm = New-Object DH3+DEVMODE
$dm.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf($dm)
[DH3]::EnumDisplaySettingsA("\\.\DISPLAY2", -1, [ref]$dm) | Out-Null
Write-Host "Current: $($dm.dmPelsWidth)x$($dm.dmPelsHeight)@$($dm.dmDisplayFrequency)Hz"

# Try to set 25Hz
$dm.dmDisplayFrequency = 25
$dm.dmFields = 0x400000  # DM_DISPLAYFREQUENCY

# Test first (CDS_TEST = 0x2)
$result = [DH3]::ChangeDisplaySettingsExA("\\.\DISPLAY2", [ref]$dm, [IntPtr]::Zero, 0x2, [IntPtr]::Zero)
Write-Host "Test 25Hz result: $result (0=success, -1=bad_mode, -2=not_updated)"

if ($result -eq 0) {
    # Apply (CDS_UPDATEREGISTRY = 0x1)
    $result2 = [DH3]::ChangeDisplaySettingsExA("\\.\DISPLAY2", [ref]$dm, [IntPtr]::Zero, 0x1, [IntPtr]::Zero)
    Write-Host "Apply 25Hz result: $result2"
} else {
    Write-Host "25Hz not supported directly. Trying 50Hz..."
    $dm.dmDisplayFrequency = 50
    $result = [DH3]::ChangeDisplaySettingsExA("\\.\DISPLAY2", [ref]$dm, [IntPtr]::Zero, 0x2, [IntPtr]::Zero)
    Write-Host "Test 50Hz result: $result"
}
