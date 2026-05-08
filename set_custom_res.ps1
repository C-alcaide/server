Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DH4 {
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

# Get current settings as base
$dm = New-Object DH4+DEVMODE
$dm.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf($dm)
[DH4]::EnumDisplaySettingsA("\\.\DISPLAY2", -1, [ref]$dm) | Out-Null
Write-Host "Current: $($dm.dmPelsWidth)x$($dm.dmPelsHeight)@$($dm.dmDisplayFrequency)Hz bpp=$($dm.dmBitsPerPel)"

# Set all relevant fields
$dm.dmPelsWidth = 1920
$dm.dmPelsHeight = 1200
$dm.dmBitsPerPel = 32
$dm.dmDisplayFrequency = 25
# DM_PELSWIDTH | DM_PELSHEIGHT | DM_BITSPERPEL | DM_DISPLAYFREQUENCY
$dm.dmFields = 0x80000 -bor 0x100000 -bor 0x40000 -bor 0x400000

# Test first
$result = [DH4]::ChangeDisplaySettingsExA("\\.\DISPLAY2", [ref]$dm, [IntPtr]::Zero, 2, [IntPtr]::Zero)
Write-Host "Test 1920x1200@25Hz: result=$result"

# Also try 50Hz
$dm.dmDisplayFrequency = 50
$result50 = [DH4]::ChangeDisplaySettingsExA("\\.\DISPLAY2", [ref]$dm, [IntPtr]::Zero, 2, [IntPtr]::Zero)
Write-Host "Test 1920x1200@50Hz: result=$result50"

# Try lower res too - 1080p@25
$dm.dmPelsWidth = 1920
$dm.dmPelsHeight = 1080
$dm.dmDisplayFrequency = 25
$result1080_25 = [DH4]::ChangeDisplaySettingsExA("\\.\DISPLAY2", [ref]$dm, [IntPtr]::Zero, 2, [IntPtr]::Zero)
Write-Host "Test 1920x1080@25Hz: result=$result1080_25"

$dm.dmDisplayFrequency = 50
$result1080_50 = [DH4]::ChangeDisplaySettingsExA("\\.\DISPLAY2", [ref]$dm, [IntPtr]::Zero, 2, [IntPtr]::Zero)
Write-Host "Test 1920x1080@50Hz: result=$result1080_50"
