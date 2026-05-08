Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public class DH8 {
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
}
"@

$dev = "\\.\DISPLAY10"
$dm = New-Object DH8+DEVMODE
$dm.dmSize = [System.Runtime.InteropServices.Marshal]::SizeOf($dm)
[DH8]::EnumDisplaySettingsA($dev, -1, [ref]$dm) | Out-Null
Write-Host "DISPLAY10 current: $($dm.dmPelsWidth)x$($dm.dmPelsHeight)@$($dm.dmDisplayFrequency)Hz"

$i = 0
$rates = @()
while ([DH8]::EnumDisplaySettingsA($dev, $i, [ref]$dm)) {
    if ($dm.dmPelsWidth -eq 1920 -and $dm.dmPelsHeight -eq 1200) {
        $rates += $dm.dmDisplayFrequency
    }
    $i++
}
$unique = $rates | Sort-Object -Unique
Write-Host "1920x1200 rates on DISPLAY10: $($unique -join ', ')"
