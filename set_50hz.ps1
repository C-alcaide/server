Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DH5 {
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

$dm = New-Object DH5+DEVMODE
$dm.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf($dm)
[DH5]::EnumDisplaySettingsA("\\.\DISPLAY2", -1, [ref]$dm) | Out-Null

$dm.dmPelsWidth = 1920
$dm.dmPelsHeight = 1080
$dm.dmBitsPerPel = 32
$dm.dmDisplayFrequency = 25
$dm.dmFields = 0x80000 -bor 0x100000 -bor 0x40000 -bor 0x400000
$r25 = [DH5]::ChangeDisplaySettingsExA("\\.\DISPLAY2", [ref]$dm, [IntPtr]::Zero, 2, [IntPtr]::Zero)
Write-Host "1920x1080@25Hz: $r25"

# Apply 1920x1080@50Hz
$dm.dmDisplayFrequency = 50
$result = [DH5]::ChangeDisplaySettingsExA("\\.\DISPLAY2", [ref]$dm, [IntPtr]::Zero, 1, [IntPtr]::Zero)
Write-Host "Apply 1920x1080@50Hz: $result (0=success)"

# Verify
[DH5]::EnumDisplaySettingsA("\\.\DISPLAY2", -1, [ref]$dm) | Out-Null
Write-Host "Now: $($dm.dmPelsWidth)x$($dm.dmPelsHeight)@$($dm.dmDisplayFrequency)Hz"
