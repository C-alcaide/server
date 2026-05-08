Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DH2 {
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

$dm = New-Object DH2+DEVMODE
$dm.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf($dm)
$rates = @()
$i = 0
while ([DH2]::EnumDisplaySettingsA("\\.\DISPLAY2", $i, [ref]$dm)) {
    if ($dm.dmPelsWidth -eq 1920 -and $dm.dmPelsHeight -eq 1200) {
        $rates += $dm.dmDisplayFrequency
    }
    $i++
}
$unique = $rates | Sort-Object -Unique
Write-Host "Available rates for 1920x1200 on DISPLAY2: $($unique -join ', ')"
